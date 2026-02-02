import argparse
import yaml
from types import SimpleNamespace
import wandb 


def load_config_from_yaml(yaml_path: str) -> SimpleNamespace:
    """Load a YAML configuration file into a Namespace-like object."""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(v) for v in d]
        else:
            return d

    return dict_to_namespace(config_dict)


def set_nested_attr(obj, key_path, value):
    """
    Set a nested attribute using dot notation.
    e.g., set_nested_attr(args, "data.roi", "V1") sets args.data.roi = "V1"
    """
    keys = key_path.split('.')
    
    # Navigate to parent object
    for key in keys[:-1]:
        if not hasattr(obj, key):
            setattr(obj, key, SimpleNamespace())
        obj = getattr(obj, key)
    
    # Set final attribute
    setattr(obj, keys[-1], value)


def parse_value(value_str):
    """Convert string value to appropriate type."""
    # Handle booleans
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    if value_str.lower() == 'none':
        return None
    
    # Handle lists (e.g., "[1,2,3]" or "1,2,3")
    if value_str.startswith('[') and value_str.endswith(']'):
        inner = value_str[1:-1]
        if inner:
            return [parse_value(v.strip()) for v in inner.split(',')]
        return []
    
    # Handle numbers
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--jobid", type=str, default=None, help="Optional job ID for wandb run")
    parser.add_argument("--override", nargs='*', default=[], help="Override config params (e.g., data.roi=V1 train.lr=0.001)")
    return parser.parse_args()


def parse_args_and_setup_wandb():
    # Parse command line arguments
    cmd_args = parse_cmd_args()
    
    # Load YAML config
    config = load_config_from_yaml(cmd_args.config)
    
    # Apply overrides with nested key support
    for override in cmd_args.override:
        if '=' not in override:
            print(f"Warning: Invalid override format '{override}', expected 'key=value'")
            continue
        
        key, value = override.split('=', 1)  # Split on first '=' only
        value = parse_value(value)
        set_nested_attr(config, key, value)
        print(f"Override: {key} = {value} (type: {type(value).__name__})")
    
    # Add jobid to config
    config.jobid = cmd_args.jobid
    config.config_path = cmd_args.config
    
    print("Job id:", config.jobid)
    
    if config.jobid is None:
        run_id = wandb.util.generate_id() 
        config.jobid = run_id
    else:
        run_id = config.jobid

    # Setup wandb
    wandb.init(
        id=run_id,
        name=run_id,
        project=config.wandb.project_name,
        config=namespace_to_dict(config),
    )
    
    return config


def namespace_to_dict(ns):
    """Convert nested SimpleNamespace back to dict (for wandb logging)."""
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    else:
        return ns