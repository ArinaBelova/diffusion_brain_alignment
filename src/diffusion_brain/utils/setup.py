import argparse
import yaml
from types import SimpleNamespace
import wandb 

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

def parse_cmd_args():
    """
    Parse command line arguments. Only requires a YAML config file.
    """
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--jobid", type=str, default=None, help="Optional job ID for wandb run")
    # parser.add_argument("--jobname", type=str, default=None, help="Optional job name for wandb run")
    args = parser.parse_args()
    return args

def parse_args_and_setup_wandb():
    # parse the config file and job_id, given as cmd arguments
    args = parse_cmd_args()
    config = load_config_from_yaml(args.config)
    args = SimpleNamespace(**vars(args), **vars(config))

    if args.jobid is None:
        run_id = wandb.util.generate_id() 
        args.jobid = run_id
    else:
        run_id = args.jobid #+ "_" + args.jobname

    # initialise wandb
    wandb.init(id=run_id, name=run_id, project=args.wandb.project_name, config=vars(args))

    return args