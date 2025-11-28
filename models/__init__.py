from models.unet import UNet
from models.dit import DiT
# for MNIST case as need to be careful with dimensions of blocks in UNet
# don't have time to think about a better way now
from diffusers import UNet1DModel, UNet2DModel
from models.gfdm_models.unet import GFDM_UNetModel
from models.mlp import ToyDiffusionMLP

def set_model(args):
    if args.model.name == "unet":
        model = UNet(args)
    elif args.model.name == "dit":
        model = DiT(args)    
    elif args.model.name == "unet-diffusers":
        model = UNet2DModel(
            in_channels=args.model.c_in,
            out_channels=args.model.c_out,
            sample_size=args.model.input_size,
            block_out_channels=(32,64,128,256),
            norm_num_groups=8,
            num_class_embeds=args.model.num_classes)
    elif args.model.name == "gfdm-unet":
        model = GFDM_UNetModel(
            image_size=args.model.input_size,
            in_channels=args.model.c_in,
            model_channels=8, # 32
            out_channels=args.model.c_out,
            num_res_blocks=1,
            attention_resolutions=(4,2),
            num_classes=args.model.num_classes,
            channel_mult=(1,2,4), # given by default but in larger resultion
            dims=1,
            dropout=0, # resnet dropout prob, not classifier-free dropout
        )
    elif args.model.name == "toy-mlp":
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