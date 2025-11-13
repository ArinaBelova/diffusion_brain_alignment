import argparse
import yaml
from types import SimpleNamespace
import torch 

from grad_updaters import set_loss_function, set_optimiser, set_learning_rate_scheduler 
from models import set_model
from data_utils import get_dataloader
import diffusivity

def load_config_from_yaml(yaml_path: str) -> SimpleNamespace:
    """
    Load a YAML configuration file into a Namespace-like object for easy dot access.
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Recursively convert dicts to Namespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(v) for v in d]
        else:
            return d

    return dict_to_namespace(config_dict)

def parse_args():
    """
    Parse command line arguments. Only requires a YAML config file.
    """
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    return args

def train_step(model, train_dataloader, loss_function, args):
    model.train()
    for _, data in enumerate(train_dataloader):
        pass
    print("Training step executed.")
    return # loss

def train(args):
    print("Training started...")
    print(f"Using the model type: {args.model.name}")

    # get all the training functions
    model = set_model(args)
    optimizer = set_optimiser(args, model)
    loss_function = set_loss_function(args)
    lr_scheduler = set_learning_rate_scheduler(optimizer, args)

    # get the dataloader
    train_dataloader, val_dataloader = get_dataloader(args)

    for epoch in range(args.train.epochs):
        print(f"Epoch {epoch+1}/{args.train.epochs} started.")
        # loss = train_step(model, train_dataloader, loss_function, args)
        # loss.backward()
        # # Clip gradient norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        # optimizer.zero_grad()
        # if lr_scheduler:
        #     lr_scheduler.step()

        # TODO: implement validation and display of generated images to wandb every M epochs 
        if epoch % args.validation.eval_freq == 0:
            print(f"Validation at epoch {epoch+1} (not implemented).")
    print("Training completed.")

if __name__ == "__main__":
    args = parse_args()
    config = load_config_from_yaml(args.config)
    args = SimpleNamespace(**vars(args), **vars(config))

    train(args)    