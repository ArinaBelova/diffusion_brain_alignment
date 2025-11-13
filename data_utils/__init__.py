from data_utils.mnist_dataloader import get_mnist_dataloader

def get_dataloader(args):
    if args.data.data_name == "mnist":
        train_dataloader, val_dataloader = get_mnist_dataloader(args)
    else:
        raise ValueError(f"Dataloader for dataset {args.data.data_name} is not yet implemented.")    
    return train_dataloader, val_dataloader