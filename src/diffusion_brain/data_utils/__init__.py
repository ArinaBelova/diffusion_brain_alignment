from diffusion_brain.data_utils.mnist_dataloader import get_mnist_dataloader
from diffusion_brain.data_utils.toy_dataloader import get_toy_dataloader
from diffusion_brain.data_utils.ann_brain_dataloader import get_ann_brain_dataloader, get_ann_brain_dataloader_multisubject

def get_dataloader(args):
    if args.data.data_name == "mnist":
        train_dataloader, val_dataloader = get_mnist_dataloader(args)
    elif args.data.data_name == "toy":
        train_dataloader, val_dataloader = get_toy_dataloader(args)
    elif args.data.data_name == "ann-brain":
        multi_subject = getattr(args.data, "multi_subject", False)
        loader_fn = get_ann_brain_dataloader_multisubject if multi_subject else get_ann_brain_dataloader
        if args.state == "train":
            train_dataloader, val_dataloader = loader_fn(args)
        else:
            val_dataloader = loader_fn(args)
            train_dataloader = None
    else:
        raise ValueError(f"Dataloader for dataset {args.data.data_name} is not yet implemented.")
    return train_dataloader, val_dataloader