from diffusion_brain.models.unet import UNet
from diffusion_brain.models.dit import DiT
from diffusion_brain.models.gfdm_models.unet import GFDM_UNetModel
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
        model = DiT(args)    
    elif args.model.name == "unet-diffusers":
        print("Setting UNet model from diffusers library")
        model = UNet2DConditionModel(
            in_channels=args.model.c_in,
            out_channels=args.model.c_out,
            sample_size=args.model.input_size,
            block_out_channels=(64,128,256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            #norm_num_groups=8,
            # cross_attention_dim=args.model.cross_attention_dim, # I don't want to have cross-attention, may come handy when I do the full project
            cross_attention_dim=args.model.cross_attention_dim,
            num_class_embeds=args.model.num_classes + 1)
        
        # playing around to figure out how to do conditioning on this model:
        # result@ don't change this class embedding!
        # model.class_embedding = torch.nn.Embedding(args.model.num_classes + 1, args.model.cross_attention_dim)

    elif args.model.name == "gfdm-unet":
        print("Setting GFDM UNet model")
        model = GFDM_UNetModel(
            image_size=args.model.input_size,
            in_channels=args.model.c_in,
            model_channels=8, # 32
            out_channels=args.model.c_out,
            num_res_blocks=1,
            attention_resolutions=(4,2),
            num_classes=args.model.num_classes + 1,
            channel_mult=(1,2,4), # given by default but in larger resultion
            dims=2, # 2 for mnist
            dropout=0, # resnet dropout prob, not classifier-free dropout
        )
    elif args.model.name == "toy-mlp":
        print("Setting Toy Diffusion MLP model")
        model = ToyDiffusionMLP(
            data_dim=args.model.input_size,
            # hidden_dim=args.model.hidden_dim,
            num_classes=args.model.num_classes,
            # num_blocks=args.model.num_blocks
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