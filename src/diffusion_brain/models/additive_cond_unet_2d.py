"""
UNet2DConditionModel subclass with additional ANN + identity conditioning.

Both signals are embedded and summed into the timestep embedding before it
reaches the ResBlocks, following the Imagen/SDXL pattern:

    emb = time_emb + ann_emb + identity_emb

ANN conditioning (continuous dense vector) uses a simple linear projection —
no Fourier features needed since the signal is already high-dimensional.
Identity conditioning (discrete class index) uses nn.Embedding.

Cross-attention from the parent class is untouched, so the ANNTokenizer →
encoder_hidden_states pathway still works as before.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


class UNet2DAdditiveConditionModel(UNet2DConditionModel):
    """UNet2DConditionModel with ANN guiding signal and subject identity
    conditioning added to the timestep embedding.

    Extra __init__ args:
        ann_dim:          dimensionality of the continuous ANN feature vector
        num_identities:   number of discrete identity classes (e.g. 8 NSD subjects)

    Extra forward kwargs:
        ann_signal:       (B, ann_dim) continuous ANN activation vector, or None
        identity_label:   (B,) integer identity indices, or None
    """

    def __init__(
        self,
        *args,
        ann_dim: int = 768,
        num_identities: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # time_embed_dim is block_out_channels[0] * 4 by default in diffusers.
        # We read it from the already-constructed time_embedding layer.
        time_embed_dim = self.time_embedding.linear_2.out_features

        # Continuous ANN conditioning: simple linear projection.
        # The ANN vector is already a dense high-dimensional representation,
        # so a learned linear map into the embedding space is sufficient.
        self.ann_embedding = nn.Linear(ann_dim, time_embed_dim)

        # Discrete subject-identity conditioning: embedding lookup table.
        # num_identities + 1 entries: indices 0..num_identities-1 are real subjects,
        # index num_identities is the null (unconditional) token for CFG dropout.
        self.identity_embedding = nn.Embedding(num_identities + 1, time_embed_dim)
        # Initialise the null token to zero so dropping identity has no effect before training
        nn.init.zeros_(self.identity_embedding.weight[num_identities])

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # --- new conditioning inputs ---
        ann_signal: Optional[torch.Tensor] = None,
        identity_label: Optional[torch.Tensor] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # ---- Reproduce the parent's embedding computation (lines 60-126) ----
        # We need to rebuild `emb` here so we can inject our extra terms
        # *before* it enters the block loop.  Everything else is delegated
        # verbatim from the parent forward.

        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        # Attention masks → bias tensors
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time embedding
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        # class embedding (parent pathway, kept for completeness)
        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        # aug embedding (parent pathway)
        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        # ---- Inject our additional conditioning into emb ----
        if ann_signal is not None:
            emb = emb + self.ann_embedding(ann_signal)

        if identity_label is not None:
            emb = emb + self.identity_embedding(identity_label)

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )

        # ---- From here on: identical to parent forward ----

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # Diffusers imports USE_PEFT_BACKEND and scale_lora_layers at module level;
        # we access them via the same path the parent does.
        from diffusers.utils import USE_PEFT_BACKEND
        if USE_PEFT_BACKEND:
            from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
            scale_lora_layers(self, lora_scale)

        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

        if is_controlnet and mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
