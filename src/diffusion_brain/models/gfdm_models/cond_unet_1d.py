import torch as th
import torch.nn as nn

from diffusion_brain.models.gfdm_models.nn import (
    timestep_embedding,
    linear,
)
from diffusion_brain.models.gfdm_models.unet import GFDM_UNetModel


class GFDM_UNet1DConditional(GFDM_UNetModel):
    """
    1D UNet with continuous conditioning.
    Conditioning is injected into the timestep embedding via a learned projection.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        cond_dim,
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
        )

        time_embed_dim = self.model_channels * 4

        # so we can add the cond embedding to time embedding
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            linear(cond_dim, time_embed_dim),
        )
        
        self.cond_dim = cond_dim
        self.downsample_factor = 2 ** (len(channel_mult) - 1)

        if len(attention_resolutions) == 0:
            # Disable the always-on middle attention block for 1D long sequences.
            self.middle_block[1] = nn.Identity()

    def forward(self, x, timesteps, cond=None):
        """
        :param x: [N, C, L]
        :param timesteps: [N]
        :param cond: [N, cond_dim] or None
        """
        if cond is None:
            cond = th.zeros(x.shape[0], self.cond_dim, device=x.device, dtype=x.dtype)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = emb + self.cond_proj(cond)

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
