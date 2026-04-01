import torch as th
import torch.nn as nn

from diffusion_brain.models.gfdm_models.nn import (
    timestep_embedding,
    linear,
)
from diffusion_brain.models.gfdm_models.unet import GFDM_UNetModel


class GFDM_UNet1DConditional(GFDM_UNetModel):
    """
    1D UNet with additive ANN conditioning:
      - ANN activations → learned projection → added to time embedding,
        modulating all residual blocks uniformly via the time-embedding pathway.
      - Time embedding slot is shared with ANN conditioning (additive).
        Future subject-identity embedding will need a separate pathway.
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

        # Additive ANN conditioning: project ANN vector into time-embedding space
        if cond_dim is not None:
            if time_embed_dim is not None:
                time_embed_dim = time_embed_dim
            else:
                time_embed_dim = model_channels * 4

            if cond_proj_type == "mlp":
                self.cond_proj = nn.Sequential(
                    nn.Linear(cond_dim, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
            else:
                # Single linear projection — no learned nonlinearity
                self.cond_proj = nn.Linear(cond_dim, time_embed_dim)
        else:
            self.cond_proj = None

    def forward(self, x, timesteps, cond=None, encoder_out=None):
        """
        :param x: [N, C, L] — 1D brain signal
        :param timesteps: [N]
        :param cond: [N, cond_dim] — raw ANN vector for additive conditioning.
            Added to time embedding to modulate all ResNet blocks.
        :param encoder_out: [N, encoder_channels, num_tokens] — tokenised ANN
            activations for cross-attention (if attention_resolutions is non-empty).
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Additive conditioning: ANN vector projected and added to time embedding
        if cond is not None and self.cond_proj is not None:
            emb = emb + self.cond_proj(cond)

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
