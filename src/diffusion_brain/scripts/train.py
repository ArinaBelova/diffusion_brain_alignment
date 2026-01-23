from types import SimpleNamespace
import torch 
import torchvision
import wandb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import itertools
import os

from diffusion_brain.utils.grad_updaters import set_loss_function, set_optimiser, set_learning_rate_scheduler 
from diffusion_brain.models import set_model
from diffusion_brain.data_utils import get_dataloader
import diffusion_brain.utils.diffusivity as diffusivity
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results

def one_step_score_estimation(x, t, noise, label, score_fn, loss_function, diffusion_process, args):
    # TODO: check this! here I simply need to estimate p_{0t}(x(t)|x(0)) mean and variance and use them to compute the true score    
    true_score = -noise
    mu, std = diffusion_process.brown_moments(x, t)
    x_t = mu + std * noise

    mask = torch.bernoulli(torch.full((len(label),), args.model.dropout_prob)).to(label.device)        
    masked_labels = label * (1 - mask) + (args.model.num_classes * mask) # don't use -1 as the empty label as nn.Embedding will throw error
    masked_labels = masked_labels.long()    

    if args.model.name == "unet-diffusers" or args.model.name == "unet-diffusers-1d":
        # class_embeddings = score_fn.class_embedding(masked_labels)  # shape [bs, cross_attn_dim]
        # class_embeddings = class_embeddings.unsqueeze(1)  # shape [bs, 1, cross_attn_dim]
        # print(f"encoder_hidden_states shape: {class_embeddings.shape}")
        # print(x_t.shape, t.shape, masked_labels.shape, encoder_hidden_states.shape)


        #######################
        t = t * 999
        #######################

        encoder_hidden_states = torch.zeros(x_t.shape[0], 1, args.model.cross_attention_dim, device=x_t.device)

        # print("masked labels before passing them to forward: ", masked_labels)
        # print("sanple: ", x_t)
        # print("time: ", t)

        predicted_score = score_fn(x_t, t, encoder_hidden_states = encoder_hidden_states, class_labels=masked_labels).sample    # encoder_hidden_states=None,
    else:
        predicted_score = score_fn(x_t, t, masked_labels)

    loss = loss_function(predicted_score, true_score)
    
    return loss

def train_step(step, model, optimizer, lr_scheduler, train_dataloader, loss_function, diffusion_process, args):
    model.train().to(DEVICE)
    # TODO: check why data type changes from float64 to DoubleTensor somewhere here...    
    data, label = next(train_dataloader)
    data = data.float().to(DEVICE)
    label = label.to(DEVICE)
    
    print(data, label)
    return 

    b, *_ = data.shape
    # sample a random timepoints for the backward process
    t = (torch.rand(b, device=data.device) * (args.diffusion.T - args.diffusion.eps) + args.diffusion.eps)

    noise = torch.randn_like(data, device=data.device)

    # run a backward SDE with this random timeline 
    loss = one_step_score_estimation(data, t, noise, label, model, loss_function, diffusion_process, args)

    # optimise the model
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    lr_scheduler.step()

    wandb.log({"train/loss": loss,
                "train/lr": lr_scheduler.get_last_lr()[0]},
                step=step)
    
def train(args):
    #torch.set_default_dtype(torch.float64)

    print("Training started...")
    print(f"Using the model type: {args.model.name}")

    # TODO: get the model if exists for continual training; probably will need to specify a path in the config file
    model = set_model(args)
    optimizer = set_optimiser(args, model)
    loss_function = set_loss_function(args)
    lr_scheduler = set_learning_rate_scheduler(optimizer, args)

    # get the dataloaders, it seems that we don't need to have a validation dataloader as we;re in the pure diffusion setting and not in bridges
    train_dataloader, _ = get_dataloader(args)
    train_dataloader = itertools.cycle(train_dataloader)

    print(f"We're getting diffusion type {args.diffusion.diffusion_type}", flush=True)
    diffusion_process = diffusivity.get_diffusion(args, device=DEVICE)

    if args.diffusion.diffusion_type == "vp":
        print(f"beta min is {diffusion_process.beta_min}, beta_max is {diffusion_process.beta_max}", flush=True)
    elif args.diffusion.diffusion_type == "ve":
        print(f"sigma_min is {diffusion_process.sigma_min}, sigma_max is {diffusion_process.sigma_max}", flush=True)

    for step in range(args.train.steps):
        print(f"Step {step+1}/{args.train.steps} started.", flush=True)
        train_step(step, model, optimizer, lr_scheduler, train_dataloader, loss_function, diffusion_process, args)
        
        if step % args.validation.eval_freq == 0:
            print(f"Validation at step {step+1}", flush=True)
            generated_samples = diffusivity.generate_samples(args.validation.batch_size, model, diffusion_process, args, device=DEVICE)
            # TODO: add other image statistics later 
            visualise_and_save_results(generated_samples, step, args)

        # save the models throughout the training
        if step % args.model.save_freq == 0 and step > 0:
            directory_to_save = f"{args.model.output_folder}/{args.jobid}"
            if not os.path.exists(directory_to_save):
                    os.makedirs(directory_to_save)
            torch.save(model.state_dict(), f"{directory_to_save}/model_step_{step}.pth")        
            print(f"Model saved at step {step}.", flush=True)    
            
    # save the model
    directory_to_save = f"{args.model.output_folder}/{args.jobid}"
    if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
    torch.save(model.state_dict(), f"{directory_to_save}/model_final.pth")     
    print("Saved the final model.", flush=True)

    print("Training completed.", flush=True)

def main():
    args = parse_args_and_setup_wandb()
    train(args)    

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()    