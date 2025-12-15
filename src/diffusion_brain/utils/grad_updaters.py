import torch
import torch.optim as optim


def set_loss_function(args):
    if args.data.data_name == "mnist" or args.data.data_name == "toy":
        loss_function = get_mnist_loss(args)
    else:
        raise ValueError(f"Loss function for dataset {args.data.data_name} is not yet implemented.")
    return loss_function    


def get_mnist_loss(args):
    if args.optim.loss == "mse":
        loss_function = torch.nn.MSELoss()
    else:
        raise ValueError(f"Loss function of type {args.loss.type} is not supported for MNIST.")
    return loss_function    


def set_optimiser(args, model):
    print(f"Setting optimizer: {args.optim.optim_name} with lr={args.optim.lr} and weight_decay={args.optim.weight_decay}")
    if args.optim.optim_name == "adamw":
        optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=args.optim.lr, weight_decay=args.optim.weight_decay)
    elif args.optim.optim_name == "adam":
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.optim.lr, weight_decay=args.optim.weight_decay)
    else:
        raise ValueError(f"Optimizer of type {args.optim.optim_name} is not supported.")

    return optimizer

def set_learning_rate_scheduler(optimizer, args):
    scheduler = None              
    
    if args.optim.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                         mode=args.optim.scheduler_mode, factor=args.optim.factor,
                                                         patience=args.optim.patience, min_lr=args.optim.min_lr)
    elif args.optim.scheduler_name == 'onecycle':
        print("Using OneCycleLR scheduler")
        print("num of total steps: ", args.train.epochs * int(args.train.dataset_size/args.train.batch_size))
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.optim.max_lr, 
                                                   total_steps=args.train.epochs * int(args.train.dataset_size/args.train.batch_size))
    
    return scheduler