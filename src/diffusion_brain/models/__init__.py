from diffusion_brain.models.unet import UNet
from diffusion_brain.models.dit import DiT
from diffusion_brain.models.gfdm_models.unet import GFDM_UNetModel
from diffusion_brain.models.gfdm_models.cond_unet_1d import GFDM_UNet1DConditional
from diffusion_brain.models.mlp import ToyDiffusionMLP

import torch
import torch.nn as nn

# for MNIST case as need to be careful with dimensions of blocks in UNet
# don't have time to think about a better way now
from diffusers import UNet1DModel, UNet2DModel, UNet2DConditionModel


class ANNTokenizer(nn.Module):
    """Learned projection from a flat ANN activation vector to multiple
    cross-attention tokens.

    With seq_len=1, cross-attention degenerates: softmax over a single element
    is always 1.0, so Q and K receive zero gradient and never learn. This
    module produces *num_tokens* > 1 tokens of dimension *token_dim*, giving
    softmax a real distribution to shape and restoring gradient flow through
    Q and K.

    The projection is a small 2-layer MLP (ann_dim → ann_dim → num_tokens *
    token_dim) trained jointly with the UNet via the denoising loss.
    """

    def __init__(self, ann_dim: int = 768, num_tokens: int = 8, token_dim: int = 256):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj = nn.Sequential(
            nn.Linear(ann_dim, ann_dim),
            nn.GELU(),
            nn.Linear(ann_dim, num_tokens * token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ann_dim) — flat ANN activation vector (already z-scored).
        Returns:
            (B, num_tokens, token_dim) — tokens for cross-attention encoder_hidden_states.
        """
        return self.proj(x).view(x.shape[0], self.num_tokens, self.token_dim)


def add_q_norm_to_cross_attn(model):
    """
    Replaces `to_q` in every attn2 (cross-attention) Attention layer with
    nn.Sequential(LayerNorm, original_to_q).

    This prevents Q-norm explosion: as training proceeds the UNet spatial
    features grow in magnitude, which saturates softmax(QK^T/sqrt(d)) and
    makes the attention output independent of K (i.e., independent of the
    conditioning signal).  Normalising Q before the projection keeps the
    dot-products in a stable range regardless of feature scale.

    The replacement is transparent to the rest of diffusers because
    Attention.forward just calls self.to_q(hidden_states).  The new
    LayerNorm parameters are proper nn.Module children so they are saved
    and loaded via state_dict automatically.

    NOTE: checkpoints saved without this patch have keys like
    '...attn2.to_q.weight'; with the patch they become
    '...attn2.to_q.1.weight' (and a new '...attn2.to_q.0.*' for the norm).
    Always apply add_q_norm_to_cross_attn() before load_state_dict() when
    loading a checkpoint that was saved with this patch.
    """
    patched = 0
    for name, module in model.named_modules():
        if "attn2" in name and type(module).__name__ == "Attention":
            if isinstance(module.to_q, nn.Sequential):
                continue  # already patched
            in_features = module.to_q.in_features
            module.to_q = nn.Sequential(
                nn.LayerNorm(in_features, elementwise_affine=True),
                module.to_q,
            )
            patched += 1
    print(f"[Q-norm] Replaced to_q with LayerNorm+Linear in {patched} attn2 layers", flush=True)
    return model


def set_model(args):
    if args.model.name == "unet":
        print("Setting UNet model")
        model = UNet(args)
    elif args.model.name == "dit":
        print("Setting DiT model")
        model = DiT(depth=args.model.depth,
                    hidden_size=args.model.hidden_size,
                    patch_size=1,
                    num_heads=args.model.num_heads,
                    input_size=args.model.input_size,
                    in_channels=args.model.c_in,
                    class_dropout_prob=args.model.dropout_prob,
                    num_classes=None,
                    label_dim=args.model.cross_attention_dim,) # for continuous labels, we can just use an MLP to embed them into the same space as timestep embeddings)    
    elif args.model.name == "unet-diffusers":
        print("Setting UNet model from diffusers library")
        # ANN conditioning goes exclusively through cross-attention (via ANNTokenizer).
        # The time-embedding / class_embed slot is deliberately left free for future
        # discrete subject-identity conditioning (nn.Embedding added to time embedding).
        unet_kwargs = dict(
            in_channels=args.model.c_in,
            out_channels=args.model.c_out,
            sample_size=args.model.input_size,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",  # CrossAttn only at deepest (smallest spatial res 64x64)
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",    # CrossAttn only at deepest
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=args.model.cross_attention_dim,
            num_class_embeds=None,
        )
        model = UNet2DConditionModel(**unet_kwargs)
        add_q_norm_to_cross_attn(model)

        # model = UNet2DConditionModel(
        #     in_channels=args.model.c_in,
        #     out_channels=args.model.c_out,
        #     sample_size=args.model.input_size,  # 256 power of 2
        #     block_out_channels=(128, 256, 512, 512),
        #     down_block_types=(
        #         "DownBlock2D",
        #         "DownBlock2D",
        #         "AttnDownBlock2D",
        #         "AttnDownBlock2D",
        #     ),
        #     up_block_types=(
        #         "AttnUpBlock2D",
        #         "AttnUpBlock2D",
        #         "UpBlock2D",
        #         "UpBlock2D",
        #     ),
        #     cross_attention_dim=args.model.cross_attention_dim,
        #     num_class_embeds=None,
        # )   
        
        # playing around to figure out how to do conditioning on this model:
        # result@ don't change this class embedding!
        # model.class_embedding = torch.nn.Embedding(args.model.num_classes + 1, args.model.cross_attention_dim)

    elif args.model.name == "gfdm-unet":
        print("Setting GFDM UNet model")
        model = GFDM_UNetModel(
            image_size=args.model.input_size,
            in_channels=args.model.c_in,
            model_channels=64, # 32
            out_channels=args.model.c_out,
            num_res_blocks=3,
            attention_resolutions=(4,2),
            num_classes=args.model.num_classes + 1,
            channel_mult=(1,2,4), # given by default but in larger resultion
            dims=2, # 2 for mnist
            dropout=0, # resnet dropout prob, not classifier-free dropout
        )
    elif args.model.name == "gfdm-unet-1d-cond":
        print("Setting GFDM UNet 1D conditional model with cross-attention")
        # ANN conditioning goes through cross-attention (via ANNTokenizer),
        # same as the 2D model. Time-embedding slot is free for subject identity.
        encoder_channels = getattr(args.model, "cross_attention_dim", 256)
        model = GFDM_UNet1DConditional(
            in_channels=args.model.c_in,
            model_channels=64,
            out_channels=args.model.c_out,
            num_res_blocks=3,
            attention_resolutions=(4,),  # cross-attention at 4× downsampled resolution
            encoder_channels=encoder_channels,
            channel_mult=(1, 2, 4),
            dropout=0,
        )
    elif args.model.name == "toy-mlp":
        print("Setting Toy Diffusion MLP model")
        model = ToyDiffusionMLP(
            data_dim=args.model.input_size,
            num_classes=args.model.num_classes,
        )        
        
    # UNet1D does not have num_class_embeds parameter    
    # elif args.model.name == "unet-diffusers-1d":
    #     model = UNet1DModel(
    #         in_channels=args.model.c_in,
    #         out_channels=args.model.c_out,
    #         sample_size=args.model.input_size,
    #         block_out_channels=(32, 32, 64), # default arg from the library
    #         norm_num_groups=8)   
    else:
        raise ValueError(f"Model of type {args.model.name} is not supported.")
    return model    
