import torch as th
import torch.nn as nn

from diffusion_brain.models.gfdm_models.nn import (
    timestep_embedding,
    linear,
)
from diffusion_brain.models.gfdm_models.unet import GFDM_UNetModel


class GFDM_UNet1DConditional(GFDM_UNetModel):
    """
    1D UNet with additive ANN + subject-identity conditioning:
      - ANN activations → learned projection → added to time embedding
      - Subject identity → nn.Embedding → added to time embedding
      - Both modulate all residual blocks uniformly via the time-embedding pathway.

        emb = time_emb + ann_emb + identity_emb
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        cond_dim=None,
        cond_proj_type="linear",
        num_identities=None,
        encoder_channels=None,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        time_embed_dim=None,
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=1,
            num_classes=None,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            encoder_channels=encoder_channels,
            time_embed_dim=time_embed_dim,
        )

        self.downsample_factor = 2 ** (len(channel_mult) - 1)

        # Resolve time_embed_dim for projection layers
        _time_embed_dim = time_embed_dim if time_embed_dim is not None else model_channels * 4

        # Additive ANN conditioning: project ANN vector into time-embedding space
        if cond_dim is not None:
            if cond_proj_type == "mlp":
                self.cond_proj = nn.Sequential(
                    nn.Linear(cond_dim, _time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(_time_embed_dim, _time_embed_dim),
                )
            else:
                # Single linear projection — no learned nonlinearity
                self.cond_proj = nn.Linear(cond_dim, _time_embed_dim)
        else:
            self.cond_proj = None

        # Discrete subject-identity conditioning: embedding lookup table
        if num_identities is not None:
            self.identity_embedding = nn.Embedding(num_identities, _time_embed_dim)
        else:
            self.identity_embedding = None

    def forward(self, x, timesteps, cond=None, encoder_out=None, identity_label=None):
        """
        :param x: [N, C, L] — 1D brain signal
        :param timesteps: [N]
        :param cond: [N, cond_dim] — raw ANN vector for additive conditioning.
            Added to time embedding to modulate all ResNet blocks.
        :param encoder_out: [N, encoder_channels, num_tokens] — tokenised ANN
            activations for cross-attention (if attention_resolutions is non-empty).
        :param identity_label: [N] — integer subject identity indices, or None.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Additive conditioning: ANN vector projected and added to time embedding
        if cond is not None and self.cond_proj is not None:
            emb = emb + self.cond_proj(cond)

        # Subject-identity conditioning
        if identity_label is not None and self.identity_embedding is not None:
            emb = emb + self.identity_embedding(identity_label)

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, encoder_out=encoder_out)
            hs.append(h)
        h = self.middle_block(h, emb, encoder_out=encoder_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, encoder_out=encoder_out)
        h = h.type(x.dtype)
        return self.out(h)
