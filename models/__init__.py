from models.unet import UNet
from models.dit import DiT

def set_model(args):
    if args.model.name == "unet":
        model = UNet(args)
    elif args.model.name == "dit":
        model = DiT(args)    
    else:
        raise ValueError(f"Model of type {args.model.name} is not supported.")
    return model    