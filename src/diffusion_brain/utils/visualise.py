import os
import torchvision
import torch
import wandb
from matplotlib import pyplot as plt
import numpy as np
import cortex
import plotly
from sklearn.metrics import r2_score

def _attach_model_step(payload, step_num):
    if step_num is None:
        return payload
    data = dict(payload)
    data["model_step"] = step_num
    return data


def visualise_and_save_results(generated_samples, step, args, step_num=None, **kwargs):
    # Use kwargs for additional arguments
    if args.data.data_name == "toy":
        generated_samples = generated_samples.cpu().numpy()

        # for toy data we need to plot scatter plots
        mean = np.mean(generated_samples, axis=0)
        print("Generated samples shape: ", generated_samples.shape)
        print("Generated samples mean:", mean)
        fig, ax = plt.subplots()

        ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6)
        
        ax.set_title(f"label {args.validation.label_to_generate} at step {step} with mean {mean[0]:.2f}, {mean[1]:.2f} and guidance_scale={args.validation.guidance_scale}")
        ax.set_xlim(-args.data.radius - 2, args.data.radius + 2)
        ax.set_ylim(-args.data.radius - 2, args.data.radius + 2)
        wandb.log(_attach_model_step({"validation_sample": wandb.Image(fig)}, step_num))
        plt.close(fig)
    elif args.data.data_name == "ann-brain":
        # for now I don't check the visual quality of generated samples, so I don't want to use this function
        # if isinstance(generated_samples, torch.Tensor):
        #     generated_samples = generated_samples[0].cpu().numpy() # take the first element of the batch for visualisation
        # else:
        #     generated_samples = generated_samples[0] # take the first element of the batch for visualisation
        # generated_samples = np.squeeze(generated_samples)
        # pyplot_brain(generated_samples, args=args, savename=f"generated_samples_step", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)

        # plot r2 correlation
        pyplot_brain(kwargs["r2_scores_across_batch_images"], args=args, savename=f"r2_scores_across_batch_images_step", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)
        r2_payload = _attach_model_step({
            "mean_r2_scores_across_batch_images": np.mean(kwargs["r2_scores_across_batch_images"]),
            "mean_r2_scores_across_voxels": np.mean(kwargs["r2_scores_across_voxels"])
        }, step_num)
        wandb.log(r2_payload)
    else:  
        generated_samples = generated_samples.cpu().numpy()  
        generated_samples = np.clip(generated_samples, 0.0, 1.0)
        grid_to_display = torchvision.utils.make_grid(generated_samples, nrow=int(np.sqrt(args.validation.batch_size)))
        wandb.log(_attach_model_step({f"validation_sample": wandb.Image(grid_to_display)}, step_num))
        # save images locally
        directory_to_save = f"{args.validation.output_folder}/{args.jobid}"
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
        torchvision.utils.save_image(generated_samples, f"{directory_to_save}/generated_samples_step_{step}.png", nrow=int(np.sqrt(args.validation.batch_size)))

# Function is courtesy of https://github.com/adriendoerig/visuo_llm/blob/main/src/nsd_visuo_semantics/utils/py_plot_brain_utils.py
def pyplot_brain(fsavg_data, savename, figpath, args, save_type='png', max_cmap_val=None, step_num=None):
    # as we work with ROI as our data, we need to reconstruct full brain data
    roi_indices_path = os.path.join(args.data.roi_defs_dir, "roi_indices", f"{args.data.roi_file}", str(args.data.roi) + ".npy")
    roi_indices = np.load(roi_indices_path, allow_pickle=True) 
    full_brain_data = np.zeros(327684)
    full_brain_data[roi_indices] = fsavg_data

    cortex.download_subject('fsaverage') 

    if max_cmap_val is None:
        boundar = np.nanmax(np.abs(fsavg_data))
    else:
        boundar = np.nanmax(np.abs(max_cmap_val))

    vert = cortex.dataset.Vertex(full_brain_data, "fsaverage", cmap='RdBu_r', vmin=-boundar, vmax=boundar)    
    flatmap = cortex.quickflat.make_figure(vert, height=480, with_colorbar=1, with_rois=False)
    
    fig = plt.gcf()

    #wandb.log({f"{savename}": wandb.Html(plotly.io.to_html(fig))})
    wandb.log(_attach_model_step({f"{savename}": wandb.Image(fig)}, step_num))

    plt.close(fig) 
    # os.makedirs(figpath, exist_ok=True) 
    # fig.suptitle(f'{savename} - max abs val: {np.nanmax(np.abs(fsavg_data)):.2f}')
    # plt.savefig(f'{figpath}/{savename}.{save_type}', dpi=600)
    # plt.close()