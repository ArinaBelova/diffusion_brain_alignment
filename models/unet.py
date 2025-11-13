
# Code taken from https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py
# TODO: for the time series data (activations & firing rates) may need Conv1d operations

# Add adaLN modules instead of MLPs to have the architecture from model alignment paper

import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_dim = args.model.time_dim
        self.remove_deep_conv = args.model.remove_deep_conv
        self.inc = DoubleConv(args.model.c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        if self.remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, args.model.c_out, kernel_size=1)

        # for conditional diffusion:
        self.conditional = args.model.num_classes is not None and args.model.dropout_prob > 0
        if self.conditional:
            # maybe add a logic for the label dropout here:
            self.label_emb = LabelEmbedder(args.model.num_classes, args.model.time_dim, args.model.dropout_prob)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t, y=None, apply_class_dropout=False):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if self.conditional:
            y = self.label_emb(y, apply_class_dropout)
        else:
            y = torch.zeros_like(t)

        c = t + y

        return self.unet_forwad(x, c)    

class LabelEmbedder(nn.Module):
    # code from M2L summer school tutorial: https://colab.research.google.com/github/M2Lschool/tutorials2025/blob/master/5_diffusion/%5BM2LS_2025%5D_Conditional_Generation_solved.ipynb#scrollTo=dba606a6
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes: int, model_dim: int, dropout_prob: float) -> None:
        super().__init__()
        use_cfg_embedding: bool = dropout_prob > 0
        self.embedding = nn.Embedding(
            num_classes + int(use_cfg_embedding),
            model_dim
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Drops labels to enable classifier-free guidance.

        :param labels: torch.Tensor of shape (B,) containing class indices.
        :return: torch.Tensor of shape (B,) with some labels possibly replaced with cfg token.
        """
        batch_size, *_ = labels.shape
        drop_ids = (
            torch.rand(
                batch_size,
                device=labels.device
            ) < self.dropout_prob
        )
        return torch.where(
            drop_ids,
            torch.full_like(
                labels,
                fill_value=self.num_classes
            ),
            labels
        )

    def forward(self, labels: torch.Tensor, should_drop: bool) -> torch.Tensor:
        """
        :param labels: torch.Tensor of shape (B,) containing class indices.
        :param should_drop: Whether to apply label dropout (usually True during training).
        :return: torch.Tensor of shape (B, model_dim) containing label embeddings.
        """
        use_dropout: bool = self.dropout_prob > 0
        if use_dropout and should_drop:
            labels = self.token_drop(labels)
        return self.embedding(labels)
    
# class UNet_conditional(UNet):
#     def __init__(self, c_in=1, c_out=1, time_dim=256, num_classes=10, dropout_prob=0.2, **kwargs):
#         super().__init__(c_in, c_out, time_dim, **kwargs)
#         self.conditional = num_classes is not None and dropout_prob > 0
#         # in conditional case:
#         if self.conditional:
#             # maybe add a logic for the label dropout here:
#             self.label_emb = LabelEmbedder(num_classes, time_dim, dropout_prob)

#     def forward(self, x, t, y=None, apply_class_dropout=False):
#         t = t.unsqueeze(-1)
#         t = self.pos_encoding(t, self.time_dim)

#         if self.conditional:
#             y = self.label_emb(y, apply_class_dropout)
#         else:
#             y = torch.zeros_like(t)

#         c = t + y        
#         # if y is not None:
#         #     # we will apply class dropout only during training; 
#         #     # during inference we want to condition on the class
#         #     t += self.label_emb(y, apply_class_dropout)

#         return self.unet_forwad(x, c)    