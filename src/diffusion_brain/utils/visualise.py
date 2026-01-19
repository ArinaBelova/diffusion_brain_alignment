import os
import torchvision
import torch
import wandb
from matplotlib import pyplot as plt
import numpy as np

def visualise_and_save_results(generated_samples, step, args):
    if args.data.data_name == "toy":
        # for toy data we need to plot scatter plots
        generated_samples = generated_samples.cpu().numpy()

        mean = np.mean(generated_samples, axis=0)
        print("Generated samples shape: ", generated_samples.shape)
        print("Generated samples mean:", mean)
        fig, ax = plt.subplots()

        ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6)
        
        ax.set_title(f"label {args.validation.label_to_generate} at step {step} with mean {mean[0]:.2f}, {mean[1]:.2f} and guidance_scale={args.validation.guidance_scale}")
        ax.set_xlim(-args.data.radius - 2, args.data.radius + 2)
        ax.set_ylim(-args.data.radius - 2, args.data.radius + 2)
        wandb.log({"validation_sample": wandb.Image(fig)}) #, step=epoch)
        plt.close(fig)
    else:    
        generated_samples = torch.clip(generated_samples, 0.0, 1.0)
        grid_to_display = torchvision.utils.make_grid(generated_samples, nrow=int(np.sqrt(args.validation.batch_size)))
        wandb.log({f"validation_sample": wandb.Image(grid_to_display)}) #, step=epoch)
        # save images locally
        directory_to_save = f"{args.validation.output_folder}/{args.jobid}"
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
        torchvision.utils.save_image(generated_samples, f"{directory_to_save}/generated_samples_step_{step}.png", nrow=int(np.sqrt(args.validation.batch_size)))