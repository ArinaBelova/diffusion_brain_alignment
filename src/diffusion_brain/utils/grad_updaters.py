import torch
import torch.optim as optim
import math

def set_loss_function(args):
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

def set_learning_rate_scheduler(optimizer, args, total_steps=None):
    """Set up learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        args: Training arguments
        total_steps: Total number of optimizer steps (for OneCycleLR with gradient accumulation)
                     If None, uses args.train.steps
    """
    scheduler = None
    
    # Use provided total_steps or fall back to args.train.steps
    num_steps = total_steps if total_steps is not None else args.train.steps
    
    if args.optim.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                         mode=args.optim.scheduler_mode, factor=args.optim.factor,
                                                         patience=args.optim.patience, min_lr=args.optim.min_lr)
    elif args.optim.scheduler_name == 'onecycle':
        print(f"Using OneCycleLR scheduler with {num_steps} total steps")
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.optim.max_lr, 
                                                   total_steps=num_steps)
    
    return scheduler