import argparse
import yaml
from types import SimpleNamespace

import torch

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

def train_step(epoch, args):
    print(f"training epoch #{epoch}")
    

def train(args):
    print("Training started...")
    print(f"Using the model type: {args.model.type}")

    for epoch in range(args.epochs):
        train_step(epoch, args)
        print(f"Epoch {epoch+1}/{args.epochs} started.")

    print("Training completed.")

if __name__ == "__main__":
    args = parse_args()
    config = load_config_from_yaml(args.config)
    args = SimpleNamespace(**vars(args), **vars(config))

    train(args)    