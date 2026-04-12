import os
import torchvision
import torch
import wandb
from matplotlib import pyplot as plt
import numpy as np
import cortex
import scipy
from diffusion_brain.utils.fmri_behav_data_utils import signal_to_2d

def _attach_model_step(payload, step_num):
    if step_num is None:
        return payload
    data = dict(payload)
    data["model_step"] = step_num
    return data


def fmri_to_wandb_image(img_2d, title=""):
    """Render a 2D fMRI map with diverging RdBu_r colormap and return a wandb.Image.

    Parameters
    ----------
    img_2d : np.ndarray
        2D array (H, W). Squeeze any leading dims before calling.
    title : str, optional
        Title for the plot.
    """
    from matplotlib.colors import TwoSlopeNorm
    img = np.squeeze(img_2d)
    abs_max = max(abs(img.min()), abs(img.max()), 1e-8)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='RdBu_r', origin="lower",
              norm=TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max))
    if title:
        ax.set_title(title)
    plt.colorbar(ax.images[0], ax=ax)
    wimg = wandb.Image(fig)
    plt.close(fig)
    return wimg


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

        return -float('inf') # for toy data we don't calculate r scores, as they are not informative for this type of data
    
    elif args.data.data_name == "ann-brain":
        # for now I don't check the visual quality of generated samples, so I don't want to use this function
        # if isinstance(generated_samples, torch.Tensor):
        #     generated_samples = generated_samples[0].cpu().numpy() # take the first element of the batch for visualisation
        # else:
        #     generated_samples = generated_samples[0] # take the first element of the batch for visualisation
        # generated_samples = np.squeeze(generated_samples)
        # pyplot_brain(generated_samples, args=args, savename=f"generated_samples_step", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)

        # plot r correlation
        if args.data.is_2d:
            r_scores_across_batch_images = get_r_across_images_2d_data(args, generated_samples, kwargs.get('true_fmri'),  model_name = step, step_num=step_num)
        else:
            r_scores_across_batch_images = get_r_across_images_1d_data(args, generated_samples, kwargs.get('true_fmri'), step_num=step_num)
            
        return r_scores_across_batch_images    

    else:  
        #generated_samples = generated_samples.cpu().numpy()  
        generated_samples = torch.clip(generated_samples, 0.0, 1.0)
        print("Generated samples properties: ", type(generated_samples), generated_samples.shape)
        grid_to_display = torchvision.utils.make_grid(generated_samples, nrow=int(np.sqrt(args.validation.batch_size)))
        wandb.log(_attach_model_step({f"validation_sample": wandb.Image(grid_to_display)}, step_num))
        # save images locally
        directory_to_save = f"{args.validation.output_folder}/{args.jobid}"
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
        torchvision.utils.save_image(generated_samples, f"{directory_to_save}/generated_samples_step_{step}.png", nrow=int(np.sqrt(args.validation.batch_size)))

        return -float('inf')

# Function is courtesy of https://github.com/adriendoerig/visuo_llm/blob/main/src/nsd_visuo_semantics/utils/py_plot_brain_utils.py
def pyplot_brain(fsavg_data, savename, figpath, args, save_type='png', max_cmap_val=None, step_num=None):
    # as we work with ROI as our data, we need to reconstruct full brain data
    roi_indices_path = os.path.join(args.data.roi_defs_dir, "roi_indices", f"{args.data.roi_file}", str(args.data.roi) + ".npy")
    roi_indices = np.load(roi_indices_path, allow_pickle=True) 
    full_brain_data = np.zeros(327684)

    if type(fsavg_data) == torch.Tensor:
        fsavg_data = fsavg_data.cpu().numpy()
        
    full_brain_data[roi_indices] = fsavg_data

    cortex.download_subject('fsaverage') 

    if max_cmap_val is None:
        boundar = np.nanmax(np.abs(fsavg_data))
    else:
        boundar = np.nanmax(np.abs(max_cmap_val))
   
    vert = cortex.dataset.Vertex(full_brain_data, "fsaverage", cmap='RdBu_r', vmin=-boundar, vmax=boundar)
    flatmap = cortex.quickflat.make_figure(vert, height=480, with_colorbar=1, with_rois=False)
    
    fig = plt.gcf()

    return fig    
    # os.makedirs(figpath, exist_ok=True) 
    # fig.suptitle(f'{savename} - max abs val: {np.nanmax(np.abs(fsavg_data)):.2f}')
    # plt.savefig(f'{figpath}/{savename}.{save_type}', dpi=600)
    # plt.close()


def get_r_across_images_2d_data(args, generated_data_roi_2d, true_fmri, step_num=None, model_name=None, **kwargs):
    if type(generated_data_roi_2d) == torch.Tensor:
        generated_data_roi_2d = generated_data_roi_2d.cpu().numpy()

    image_shape = true_fmri.squeeze(1).shape[1:]  # Assuming true_fmri has shape (num_images, H, W) or (num_images, 1, H, W)

    print("Generated data shape (before processing):", generated_data_roi_2d.shape)  # Should be (num_images, H, W)
    locations_load_path = os.path.join(
            args.data.roi_defs_dir, f"roi_preselected_extended_2d_images_res_{args.data.grid_resolution_2d}", 
            f"{args.data.roi_file}",
            f"{args.data.subj}_{args.data.roi}.npz"
        )
    
    locations_roi = np.load(locations_load_path, allow_pickle=True)["locations"]  # [2, n_locations]

    print("True signal shape: ", true_fmri.shape)  # Should be (n_locations, n_timepoints)

    # Plot generated and true fMRI with the same RdBu_r colorscheme as the r-score map
    n_preview = min(3, generated_data_roi_2d.shape[0])

    # In multi-subject mode the pooled tensor is the concatenation of all 8
    # subjects' test slices, so the first few indices happen to be subj01 — tag
    # the preview titles so the reader knows these are pooled-order previews
    # and the per-subject panels (logged separately from generate.py) are the
    # authoritative per-subject views.
    is_multi_subj = getattr(args.data, "multi_subject", False)
    preview_prefix = "Pooled " if is_multi_subj else ""

    generated_data_to_report_wandb = [
        fmri_to_wandb_image(generated_data_roi_2d[i], title=f"{preview_prefix}Generated sample {i}")
        for i in range(n_preview)
    ]

    true_fmri_to_report_wandb = [
        fmri_to_wandb_image(true_fmri[i], title=f"{preview_prefix}True fMRI sample {i}")
        for i in range(n_preview)
    ]

    y_coords = locations_roi[0]
    x_coords = locations_roi[1]

    if len(generated_data_roi_2d.shape) > 2:        
        # Extract time series for each valid location: shape (num_images, n_locations)
        time_series = generated_data_roi_2d[:, y_coords, x_coords]
        print("Time series shape:", time_series.shape)  # Should be (num_images, n_locations)
    else:
        time_series = generated_data_roi_2d    

    if len(true_fmri.shape) > 2:
        true_fmri = true_fmri.squeeze(1)[:, y_coords, x_coords]
        print("True signal shape after squeezing:", true_fmri.shape)  # Should be (grid_size, grid_size)

    if type(true_fmri) == torch.Tensor:
        true_fmri = true_fmri.cpu().numpy()
        
    # Calculate Pearson correlation for each location with either mean signal or with true signal
    r_arr = np.array([
        scipy.stats.pearsonr(time_series[:, i], true_fmri[:, i])[0]
        for i in range(len(x_coords))
    ])

    print("r_arr shape:", r_arr.shape)  # Should be (n_locations,)

    #r_img = np.zeros_like(generated_data_roi_2d[0])  # Assuming grid_size is the same for x and y
    r_img = np.zeros(image_shape)

    print("r_img shape:", r_img.shape)  # Should be (grid_size, grid_size)

    r_img[y_coords, x_coords] = r_arr

    from matplotlib.colors import TwoSlopeNorm
    abs_max = max(abs(r_img.min()), abs(r_img.max()))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    #norm = TwoSlopeNorm(vmin=r_img.min(), vcenter=0, vmax=r_img.max())

    plt.imshow(r_img, cmap='RdBu_r', origin="lower", norm=norm)
    plt.colorbar()
    pool_suffix = " (pooled across subjects)" if is_multi_subj else ""
    plt.title(
        f"Correlation (r) across images for each location{pool_suffix}, "
        f"resolution={args.data.grid_resolution_2d}mm"
    )
    fig = plt.gcf()

    print("Calculating r scores across voxels...", flush=True)
    r_scores_across_voxels = np.empty((time_series.shape[0],))
    r_scores_across_batch_images = np.empty((time_series.shape[1],))
    print("Shape of r_scores_across_voxels array:", r_scores_across_voxels.shape)  # Should be (num_images,)

    print("Calculating r scores across batch images...", flush=True)
    for voxel_idx in range(time_series.shape[1]):
        true_fmri_by_image = true_fmri[:,voxel_idx]
        generated_sample = time_series[:,voxel_idx]
        assert len(true_fmri_by_image) == len(generated_sample)
        r_scores_across_batch_images[voxel_idx] = scipy.stats.pearsonr(true_fmri_by_image, generated_sample)[0]

    print("Calculating r scores across voxels...", flush=True)
    for image_idx in range(time_series.shape[0]):
        true_fmri_by_voxel = true_fmri[image_idx,:] 
        generated_sample = time_series[image_idx,:]
        assert len(true_fmri_by_voxel) == len(generated_sample)
        r_scores_across_voxels[image_idx] = scipy.stats.pearsonr(true_fmri_by_voxel, generated_sample)[0]

    mse = np.mean((time_series - true_fmri) ** 2)

    r_payload = _attach_model_step({f"generated_data": generated_data_to_report_wandb,
                                    f"true_fmri_data": true_fmri_to_report_wandb,
                                    "r_image": wandb.Image(fig),
                                    "mean_r_scores_across_voxels": np.mean(r_scores_across_voxels),
                                    "mean_r_scores_across_batch_images": np.mean(r_scores_across_batch_images),
                                    "mse": mse}, step_num)
    wandb.log(r_payload)

    plt.close(fig)    

    return np.mean(r_scores_across_batch_images) # important for saving the best running model based on this metric 

def get_r_across_images_1d_data(args, generated_data_roi, true_fmri, step_num=None, **kwargs):
    
    # Generate and log 3 samples of predicted and true fMRI data
    # Build the grid
    figs_pred, figs_true = [], []

    if type(generated_data_roi) == torch.Tensor:
        generated_data_roi = generated_data_roi.cpu().numpy()

    if type(true_fmri) == torch.Tensor:
        true_fmri = true_fmri.cpu().numpy()

    for i in range(min(3, len(generated_data_roi))):
        print("fmri_predicted shape is ", generated_data_roi[i].shape)
        fig_pred = pyplot_brain(generated_data_roi[i], args=args, savename=f"generated_sample_idx_{i}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')
        figs_pred.append(fig_pred)        
        print("true_fmri_dataset shape is ", true_fmri[i].shape)
        fig_true = pyplot_brain(true_fmri[i], args=args, savename=f"true_sample_idx_{i}_step", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')
        figs_true.append(fig_true)
    
    images_pred = [wandb.Image(f) for f in figs_pred]
    images_true = [wandb.Image(f) for f in figs_true]

    # 2D flatmap projections of the same samples
    n_preview = min(3, len(generated_data_roi))
    images_pred_2d, images_true_2d = [], []
    for i in range(n_preview):
        gen_i = generated_data_roi[i]
        true_i = true_fmri[i]
        gen_2d, _ = signal_to_2d(args, one_signal_to_transform=gen_i)
        true_2d, _ = signal_to_2d(args, one_signal_to_transform=true_i)
        images_pred_2d.append(fmri_to_wandb_image(gen_2d[0], title=f"Generated 2D sample {i}"))
        images_true_2d.append(fmri_to_wandb_image(true_2d[0], title=f"True fMRI 2D sample {i}"))

    # to store r scores across images for each voxel
    r_scores_across_batch_images = np.empty((generated_data_roi.shape[1],))
    r_scores_across_voxels = np.empty((generated_data_roi.shape[0],))

    print("True fmri shape:", true_fmri.shape, flush=True)
    print("Generated samples shape:", generated_data_roi.shape, flush=True)

    # TODO: think how interoduce inter-voxels statistics as here we treat all the voxels independently and calculate correlation across images for each voxel separately, 
    # but maybe we can also look at the correlation across voxels not to fall back to the univariate methods approaches
    print("Calculating r scores across batch images...", flush=True)
    for voxel_idx in range(generated_data_roi.shape[1]):
        true_fmri_by_image = true_fmri[:,voxel_idx]
        generated_sample = generated_data_roi[:,voxel_idx]
        assert len(true_fmri_by_image) == len(generated_sample)
        r_scores_across_batch_images[voxel_idx] = scipy.stats.pearsonr(true_fmri_by_image, generated_sample)[0]

    print("Calculating r scores across voxels...", flush=True)
    for image_idx in range(generated_data_roi.shape[0]):
        true_fmri_by_voxel = true_fmri[image_idx,:]
        generated_sample = generated_data_roi[image_idx,:]
        assert len(true_fmri_by_voxel) == len(generated_sample)
        r_scores_across_voxels[image_idx] = scipy.stats.pearsonr(true_fmri_by_voxel, generated_sample)[0]
    
    r_scores_2d, _ = signal_to_2d(args, one_signal_to_transform=r_scores_across_batch_images)
    r_scores_2d_wandb = fmri_to_wandb_image(r_scores_2d[0], title="r-scores across images (2D)")
    fig_r_scores_across_img = pyplot_brain(r_scores_across_batch_images, args=args, savename=f"r_scores_across_batch_images_step", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)
    fig_generated_data = pyplot_brain(generated_data_roi.mean(axis=0), args=args, savename=f"generated_data_mean_across_images_step", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)
    fig_true_fmri = pyplot_brain(true_fmri.mean(axis=0), args=args, savename=f"true_fmri_mean_across_images_step", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)

    mse = np.mean((generated_data_roi - true_fmri) ** 2)

    r_payload = _attach_model_step({
        "three_generated_images": images_pred,
        "three_true_fmri_images": images_true,
        "three_generated_images_2d": images_pred_2d,
        "three_true_fmri_images_2d": images_true_2d,
        "r_scores_across_batch_images": wandb.Image(fig_r_scores_across_img),
        "r_scores_across_batch_images_2d": r_scores_2d_wandb,
        "mean_generated_data_across_batch_images": wandb.Image(fig_generated_data),
        "mean_true_fmri_data_across_batch_images": wandb.Image(fig_true_fmri),
        "mean_r_scores_across_batch_images": np.mean(r_scores_across_batch_images),
        "mean_r_scores_across_voxels": np.mean(r_scores_across_voxels),
        "mse": mse
    }, step_num)

    wandb.log(r_payload)

    return np.mean(r_scores_across_batch_images)

                    