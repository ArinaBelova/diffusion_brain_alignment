import torch
import torch.optim as optim
import math


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    Standard practice for diffusion model training (Nichol & Dhariwal, 2021;
    Song et al., 2021). The EMA weights are used for evaluation/generation
    while the original weights continue to be optimised.

    Supports optional warmup: during early training the effective decay ramps
    from 0 (i.e. shadow == current params) up to *decay* over *warmup_steps*
    optimiser updates.  This prevents the shadow from being anchored to the
    random initialisation.
    """

    def __init__(self, parameters, decay=0.9999, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = [p.clone().detach() for p in parameters]

    def _current_decay(self):
        if self.warmup_steps > 0 and self.num_updates < self.warmup_steps:
            return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        return self.decay

    @torch.no_grad()
    def update(self, parameters):
        decay = self._current_decay()
        for s, p in zip(self.shadow, parameters):
            if p.requires_grad:
                s.mul_(decay).add_(p.data, alpha=1 - decay)
        self.num_updates += 1

    def copy_to(self, parameters):
        """Copy shadow parameters into model parameters (for evaluation)."""
        for s, p in zip(self.shadow, parameters):
            p.data.copy_(s)

    def store(self, parameters):
        """Backup current model parameters before swapping in EMA weights."""
        self.backup = [p.data.clone() for p in parameters]

    def restore(self, parameters):
        """Restore original model parameters after evaluation."""
        for b, p in zip(self.backup, parameters):
            p.data.copy_(b)
        self.backup = None

    def state_dict(self):
        return {
            "shadow": self.shadow,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.warmup_steps = state_dict.get("warmup_steps", 0)


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