from diffusion_brain.models.unet import UNet
from diffusion_brain.models.dit import DiT
from diffusion_brain.models.gfdm_models.unet import GFDM_UNetModel
from diffusion_brain.models.gfdm_models.cond_unet_1d import GFDM_UNet1DConditional
from diffusion_brain.models.mlp import ToyDiffusionMLP

import torch 

# for MNIST case as need to be careful with dimensions of blocks in UNet
# don't have time to think about a better way now
from diffusers import UNet1DModel, UNet2DModel, UNet2DConditionModel

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
        # Use CrossAttn only at deepest layer (smallest resolution) to save memory
        model = UNet2DConditionModel(
            in_channels=args.model.c_in,
            out_channels=args.model.c_out,
            sample_size=args.model.input_size,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",  # CrossAttn only at deepest (32x32 for 256 input)
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",    # CrossAttn only at deepest
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=args.model.cross_attention_dim,
            num_class_embeds=None,
        )
        
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
        print("Setting GFDM UNet 1D conditional model")
        model = GFDM_UNet1DConditional(
            in_channels=args.model.c_in,
            model_channels=64,
            out_channels=args.model.c_out,
            num_res_blocks=3,
            attention_resolutions=(), # (4, 2)
            cond_dim=args.model.cross_attention_dim,
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
