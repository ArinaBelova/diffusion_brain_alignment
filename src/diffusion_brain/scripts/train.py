import argparse
import yaml
from types import SimpleNamespace
import torch 
import torchvision
import wandb
import numpy as np
import matplotlib.pyplot as plt
import datetime

from diffusion_brain.utils.grad_updaters import set_loss_function, set_optimiser, set_learning_rate_scheduler 
from diffusion_brain.models import set_model
from diffusion_brain.data_utils import get_dataloader
import diffusion_brain.utils.diffusivity as diffusivity

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
    parser.add_argument("--jobid", type=str, default=None, help="Optional job ID for wandb run")
    args = parser.parse_args()
    return args

def one_step_score_estimation(x, t, noise, label, score_fn, loss_function, args):
    #t = t[:, None, None, None] # to match the datapoint dimensions for the further arithmetic; NOTE: in the model we need to have t as (b,)

    # TODO: check this! here I simply need to estimate p_{0t}(x(t)|x(0)) mean and variance and use them to compute the true score
    true_score = -noise
    
    if args.model.name == "unet-diffusers" or args.model.name == "unet-diffusers-1d":
        print(x.shape, noise.shape, label.shape)
        predicted_score = score_fn(x, t * 1000, label, return_dict=False)[0] # multiply by 1000 so time embedding works better
    elif args.model.name == "unet":    
        # in training we would like to drop classes sometimes for classifier-free conditioning
        predicted_score = score_fn(x, t * 1000, label, apply_class_dropout=True) # multiply by 1000 so time embedding works better
    else:
        # toy-mlp is under this case as we're explicitly mask the label here:
        mask = torch.bernoulli(torch.full((len(label),), args.model.dropout_prob)).to(label.device)
        
        #print(label.shape, mask.shape, (args.model.num_classes * mask).shape)
        masked_labels = label * (1 - mask) + (args.model.num_classes * mask)
        masked_labels = masked_labels.long()
        predicted_score = score_fn(x, t * 1000, masked_labels)

    #print(f"predicted score {predicted_score} \t \t \t true score {true_score}")
    loss = loss_function(predicted_score, true_score)
 
    return loss

def train_epoch(epoch, model, optimizer, lr_scheduler, train_dataloader, loss_function, args)-> torch.Tensor:
    model.train().to(DEVICE)
    avg_loss = 0.0

    for idx, (data, label) in enumerate(train_dataloader):
        ##### visualise training data, remove later ######
        # print("Visualising training data samples...")
        # print(f"created train dataloader is {train_dataloader}")
        # print(f"in training datashape is {data.shape}")
        # visualise_results(data, epoch, args)
        #data = data.to(DEVICE)
        ###############################################

        # TODO: check why data type changes from float64 to DoubleTensor somewhere here...
        data = data.float().to(DEVICE)
        label = label.to(DEVICE) # as label is given by default as (b, 1)

        b, *_ = data.shape
        # sample a random timepoints for the backward process
        #t = torch.rand((b, ), device=data.device)
        t = (torch.rand(b, device=data.device) * (args.diffusion.T - args.diffusion.eps) + args.diffusion.eps)
        noise = torch.randn_like(data, device=data.device)

        # run a backward SDE with this random timeline 
        loss = one_step_score_estimation(data, t, noise, label, model, loss_function, args)
        
        avg_loss += loss.item()
        step = epoch * len(train_dataloader) + idx

        # optimise the model
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        # TODO: check how lr scheduler is usually updated and whether we do 1 additional step 
        #print("step #{}, loss: {:.6f}".format(step, loss.item()))
        lr_scheduler.step()
        wandb.log({"train/loss": loss,
                   "train/lr": lr_scheduler.get_last_lr()[0]},
                    step=step)
    
    # report average loss of current epoch to wandb:
    avg_loss /= len(train_dataloader)

    return avg_loss

def visualise_results(generated_samples, epoch, args):
    if args.data.data_name == "toy":
        # for toy data we need to plot scatter plots
        generated_samples = generated_samples.cpu().numpy()
        fig, ax = plt.subplots()

        ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6)
        
        ax.set_title(f"Generated Samples label {args.validation.label_to_generate} at Epoch {epoch}")
        wandb.log({"validation_sample": wandb.Image(fig)}) #, step=epoch)
        plt.close(fig)
    else:    
        grid_to_display = torchvision.utils.make_grid(generated_samples, nrow=np.sqrt(args.validation.batch_size))
        wandb.log({"validation_sample": wandb.Image(grid_to_display)}) #, step=epoch)
    
def train(args):
    #torch.set_default_dtype(torch.float64)

    print("Training started...")
    print(f"Using the model type: {args.model.name}")

    # get all the training functions
    model = set_model(args)
    optimizer = set_optimiser(args, model)
    loss_function = set_loss_function(args)
    lr_scheduler = set_learning_rate_scheduler(optimizer, args)

    # get the dataloaders, it seems that we don't need to have a validation dataloader as we;re in the pure diffusion setting and not in bridges
    train_dataloader, _ = get_dataloader(args)

    print(f"We're getting diffusion type {args.diffusion.diffusion_type}")
    diffusion_process = diffusivity.get_diffusion(args, device=DEVICE)
    print(f"beta min is {diffusion_process.beta_min}, beta_max is {diffusion_process.beta_max}")

    for epoch in range(args.train.epochs):
        print(f"Epoch {epoch+1}/{args.train.epochs} started.")
        avg_epoch_loss = train_epoch(epoch, model, optimizer, lr_scheduler, train_dataloader, loss_function, args)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}")
        wandb.log({"train/avg_epoch_loss": avg_epoch_loss}) #, step=epoch)
        
        # TODO: implement validation and display of generated images to wandb every eval_freq epochs 
        if epoch % args.validation.eval_freq == 0:
            print(f"Validation at epoch {epoch+1}")
            generated_samples = diffusivity.generate_samples(args.validation.batch_size, model, diffusion_process, args, device=DEVICE)
            # TODO: add other image statistics later 
            visualise_results(generated_samples, epoch, args)
            
    print("Training completed.")

def main():
    # parse the config file and job_id, given as cmd arguments
    args = parse_args()
    config = load_config_from_yaml(args.config)
    args = SimpleNamespace(**vars(args), **vars(config))

    if args.jobid is None:
        run_id = wandb.util.generate_id() # + f"-g-{args.diffusion.max_diffusivity}" # + f"-K-{args.K}" + f"-H-{args.H}" + f"-norm-{args.norm}"
    else:
        run_id = args.jobid # + datetime.datetime.now().strftime("d %D h %H m %M") # + f"-g-{args.diffusion.max_diffusivity}"

    # initialise wandb
    wandb.init(id=run_id, name=run_id, project=args.wandb.project_name, config=vars(args))

    train(args)    

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()    