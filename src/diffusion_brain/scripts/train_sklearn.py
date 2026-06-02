from sklearn.linear_model import RidgeCV
import torch
from scipy.stats import pearsonr
import numpy as np
# import rsatoolbox # need to install it again in the cluster
import wandb
import os
#from fracridge import FracRidgeRegressorCV

from diffusion_brain.data_utils import get_dataloader
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
# from diffusion_brain.utils.fmri_behav_data_utils import compute_rdm
from diffusion_brain.utils.visualise import pyplot_brain, get_r_across_images_2d_data, get_r_across_images_1d_data, fmri_to_wandb_image
from diffusion_brain.utils.nc_correction import (
    NC_MASK_KEY_DEFAULT,
    NC_MASKS_PATH_DEFAULT,
    build_brain_map,
    correct_r_by_nc_with_mask,
    get_nc_perm_aligned,
    per_voxel_r,
)

def get_train_test_numpy_datasets(train_dataloader, test_dataloader, args):
    print(f"Length of the dataloaders: {len(train_dataloader)}, {len(test_dataloader)}", flush=True)
    # 2. Extract the single large batch
    train_fmri_dataset, train_activations_dataset = next(iter(train_dataloader))
    test_fmri_dataset, test_activations_dataset = next(iter(test_dataloader))

    return train_fmri_dataset.numpy(), test_fmri_dataset.numpy(),train_activations_dataset.numpy(), test_activations_dataset.numpy()

def preprocess_2d_dataset(dataset, args):
    # reshape the 2D fmri data to 1D (flatten the spatial dimensions)
    print("Dataset shape (before processing):", dataset.shape)  # Should be (num_images, H, W)
    locations_load_path = os.path.join(
            args.data.roi_defs_dir, f"roi_preselected_extended_2d_images_res_{args.data.grid_resolution_2d}", 
            f"{args.data.roi_file}",
            f"{args.data.subj}_{args.data.roi}.npz"
        )
    
    locations_roi = np.load(locations_load_path, allow_pickle=True)["locations"]  # [2, n_locations]
    y_coords = locations_roi[0]
    x_coords = locations_roi[1]
    
    dataset = dataset.squeeze(1)  # shape (num_images, H, W)
    dataset = dataset[:, y_coords, x_coords]  # shape (num_images, n_locations)
    print("Dataset shape after selecting ROI locations: ", dataset.shape, flush=True)
    
    return dataset, locations_roi

def train(train_activations_dataset, train_fmri_dataset, args):
    print("Training Ridge Regression with Cross-Validation...", flush=True)

    if args.data.is_2d:
        train_fmri_dataset, _ = preprocess_2d_dataset(train_fmri_dataset, args)

    print("Stats of train activations dataset:", train_activations_dataset.mean(), train_activations_dataset.std(), flush=True)
    print("Stats of train fMRI dataset:", train_fmri_dataset.mean(), train_fmri_dataset.std(), flush=True)

    #alphas = np.logspace(-2, 6, 9)
    alphas = np.array([1e-3, 1e-2, 1e-1, 1])

    #################### DEBUG WITH PURE NOISE DATA ###################
    #train_activations_dataset = np.random.randn(*train_activations_dataset.shape)
    # train_fmri_dataset = np.random.randn(*train_fmri_dataset.shape)

    # permute rows (images) in activations matrix
    # np.random.shuffle(train_activations_dataset)
    ##############################################
    print("Train activations dataset shape: ", train_activations_dataset.shape, flush=True)
    print("Train fMRI dataset shape: ", train_fmri_dataset.shape, flush=True)

    if args.model.simple_ridge:
        clf = RidgeCV(alphas=alphas, scoring="r2").fit(train_activations_dataset, train_fmri_dataset)
    else:
        from sklearn.metrics import make_scorer

        def pearson_scorer(y_true, y_pred):
            """Mean Pearson r across voxels (or single voxel)."""
            if y_true.ndim == 1:
                return pearsonr(y_true, y_pred)[0]
            # For multi-output: mean correlation across voxels
            rs = [pearsonr(y_true[:, i], y_pred[:, i])[0] 
                for i in range(y_true.shape[1])]
            return np.mean(rs)

        pearson_scoring = make_scorer(pearson_scorer)

        clf = RidgeCV(
            alphas=alphas,
            scoring=pearson_scoring,
            alpha_per_target=True
        )
        clf.fit(train_activations_dataset, train_fmri_dataset)
    
    # , scoring="r2"
    print(f"Scoring used: {clf.scoring}")
    # clf = FracRidgeRegressorCV().fit(train_activations_dataset, train_fmri_dataset, frac_grid=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    # print("Learnt best frac parameter", clf.best_frac_, flush=True)
    print("Learn features shape: ", clf.coef_.shape, flush=True)
    print("Learned features stats: ", clf.coef_.mean(), clf.coef_.std(), flush=True)
    
    print("Training completed.", flush=True)
    print("Evaluating on training data...", flush=True)
    
    final_score = clf.score(train_activations_dataset, train_fmri_dataset)
    # Also compute Pearson r per voxel (same metric as diffusion model evaluation)
    train_pred = clf.predict(train_activations_dataset)
    train_r_per_voxel = np.array([
        pearsonr(train_fmri_dataset[:, i], train_pred[:, i])[0]
        for i in range(train_fmri_dataset.shape[1])
    ])
    print("Evaluation completed.", flush=True)
    print(f"Best Alpha and length of alphas: {clf.alpha_}, {len(clf.alpha_)}", flush=True)
    print(f"Train R^2 (sklearn, flattened): {final_score}", flush=True)
    print(f"Train mean Pearson r (per voxel): {np.mean(train_r_per_voxel):.4f}", flush=True)

    return clf

def _apply_within_subject_nc_correction(
    fmri_predicted_1d, true_fmri_1d, args, y_coords, x_coords, image_shape, step,
):
    """Apply within-subject permutation NC + sig-mask correction and log to wandb.

    Mirrors `_evaluate_single_subject_nc_correction` in generate.py: divides
    per-voxel r by the within-subject NC lower bound for voxels passing the
    significance mask (loaded from the aggregated permutation pkl). Voxels
    failing the mask are excluded from the corrected metric.
    """
    subj = args.data.subj
    masks_path = getattr(args.data, "nc_masks_path", NC_MASKS_PATH_DEFAULT)
    mask_key = getattr(args.validation, "nc_mask_key", NC_MASK_KEY_DEFAULT)

    nc_pixels, sig_mask, nc_source = get_nc_perm_aligned(
        subj, args.data.roi_file, args.data.roi, y_coords, x_coords,
        masks_path=masks_path, mask_key=mask_key,
    )
    if nc_pixels is None:
        print(
            f"\nWARNING: within-subject permutation NC entry not found for "
            f"({subj}, {args.data.roi_file}_{args.data.roi}) at {masks_path} — "
            "skipping NC correction.",
            flush=True,
        )
        return

    r_per_voxel = per_voxel_r(fmri_predicted_1d, true_fmri_1d)
    r_corrected, apply_mask = correct_r_by_nc_with_mask(r_per_voxel, nc_pixels, sig_mask)
    n_apply = int(apply_mask.sum())
    n_total = len(nc_pixels)
    raw_mean = float(np.nanmean(r_per_voxel))
    nc_mean = float(np.nanmean(nc_pixels))
    mean_rc = float(np.nanmean(r_corrected))
    median_rc = float(np.nanmedian(r_corrected))

    print(f"\n{'=' * 60}", flush=True)
    print(f"Single-subject NC correction (Ridge baseline, {subj})", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  NC source              : {nc_source}", flush=True)
    print(
        f"  significant voxels (applied): {n_apply}/{n_total} "
        f"({100.0 * n_apply / max(n_total, 1):.1f}%)",
        flush=True,
    )
    print(f"  raw mean r              = {raw_mean:+.4f}", flush=True)
    print(f"  mean NC lower bound     = {nc_mean:+.4f}", flush=True)
    print(f"  mean r/NC (corrected)   = {mean_rc:+.4f}", flush=True)
    print(f"  median r/NC (corrected) = {median_rc:+.4f}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    rc_img = build_brain_map(r_corrected, image_shape, y_coords, x_coords)
    nc_img = build_brain_map(nc_pixels, image_shape, y_coords, x_coords)
    sig_img = build_brain_map(sig_mask.astype(np.float64), image_shape, y_coords, x_coords)

    wandb.log({
        "noise_corrected/mean_r": mean_rc,
        "noise_corrected/median_r": median_rc,
        "noise_corrected/raw_mean_r": raw_mean,
        "noise_corrected/mean_nc_lower_bound": nc_mean,
        "noise_corrected/n_apply_voxels": n_apply,
        "noise_corrected/n_total_voxels": n_total,
        "noise_corrected/frac_apply": float(n_apply / max(n_total, 1)),
        "noise_corrected/nc_source": nc_source,
        "noise_corrected/r_image": fmri_to_wandb_image(
            rc_img, title=f"{subj} r/NC (Ridge, step={step})", robust=True,
        ),
        "noise_corrected/nc_lower_bound_image": fmri_to_wandb_image(
            nc_img,
            title=f"NC lower bound ({subj}) — {args.data.roi_file}_{args.data.roi}",
        ),
        "noise_corrected/sig_mask_image": fmri_to_wandb_image(
            sig_img, title=f"{subj} within-subject sig mask",
        ),
        "model_step": step,
    })

    output_dir = os.path.join(
        getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
        str(args.jobid),
    )
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "ridge_single_subject_nc_corrected.npz"),
        r_per_voxel=r_per_voxel,
        r_corrected=r_corrected,
        nc_lower_bound=nc_pixels,
        sig_mask=sig_mask,
        apply_mask=apply_mask,
        nc_source=nc_source,
        subj=subj,
        roi=args.data.roi,
        roi_file=args.data.roi_file,
    )


def validate_and_visualise(clf, true_activations_dataset, true_fmri_dataset, args, step="sklearn_fitting"):
    true_fmri_dataset_copy = true_fmri_dataset.copy()  # Make a copy to avoid modifying the original dataset
    if args.data.is_2d:
        # reshape the 2D fmri data to 1D (flatten the spatial dimensions)
        true_fmri_dataset, locations_roi = preprocess_2d_dataset(true_fmri_dataset, args)
        
    # validate and display the results
    print("Generating predicted fMRI data from activations...", flush=True)
    print("Shape of activations dataset:", true_activations_dataset.shape, flush=True)

    fmri_predicted = clf.predict(true_activations_dataset)  # handles intercept correctly for both modes
    if args.model.simple_ridge:
        score_test = clf.score(true_activations_dataset, true_fmri_dataset)
        print("R^2 score on test data: ", score_test, flush=True)
    else:
        ss_res = np.sum((true_fmri_dataset - fmri_predicted) ** 2, axis=0)
        ss_tot = np.sum((true_fmri_dataset - true_fmri_dataset.mean(axis=0)) ** 2, axis=0)
        r2_scores = 1 - ss_res / ss_tot
        print("Mean R^2 score across voxels on test data: ", np.mean(r2_scores), flush=True)

    ######## 1D Visualisation of predicted and true fMRI data for the test set ##########
    if args.data.is_2d:
        # transform the predicted 1D fMRI data back to 2D format for visualisation
        print("Transforming predicted fMRI data back to 2D format for visualifmri_predictedsation...", flush=True)
        # true_fmri_2d = true_fmri_dataset_copy.squeeze(1)  # (num_images, H, W) — drop channel dim
        fmri_predicted_2d = np.zeros_like(true_fmri_dataset_copy.squeeze(1))
        y_coords = locations_roi[0]
        x_coords = locations_roi[1]
        fmri_predicted_2d[:, y_coords, x_coords] = fmri_predicted  # fill in the predicted values at the ROI locations
        get_r_across_images_2d_data(args, fmri_predicted_2d, true_fmri_dataset_copy, step_num=step)

        # Within-subject NC correction (mirrors generate.py single-subject path)
        _apply_within_subject_nc_correction(
            fmri_predicted_1d=fmri_predicted,
            true_fmri_1d=true_fmri_dataset,
            args=args,
            y_coords=y_coords,
            x_coords=x_coords,
            image_shape=fmri_predicted_2d.shape[1:],
            step=step,
        )
        fmri_predicted = fmri_predicted_2d
    else:
        print("Visualising predicted fMRI data...", flush=True)
        # can only visualise one batch element
        pyplot_brain(fmri_predicted[0], args=args, savename=f"generated_sample_idx_{0}_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

        print("Visualising true fMRI data...", flush=True)
        pyplot_brain(true_fmri_dataset[0], args=args, savename=f"true_sample_idx_{0}_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

        print("Visualising the difference MSE between true and predicted fMRI data...", flush=True)
        print("Shape of true fMRI dataset:", true_fmri_dataset.shape, flush=True)
        difference = (true_fmri_dataset - fmri_predicted)**2
        pyplot_brain(difference.mean(axis=0), args=args, savename=f"roi_{args.data.roi}_difference_mse_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

        get_r_across_images_1d_data(args, fmri_predicted, true_fmri_dataset, step_num=step)
    ########## RDM visualisation on test dataset; need to update the container for that ##########
    # print("Visualising RDM on test dataset: ", flush=True)
    # rdms_true_test_fmri = compute_rdm(true_fmri_dataset, args, regime="test")
    # rdms_predicted_test_fmri = compute_rdm(fmri_predicted, args, regime="test")

    # wandb.log({"rdm_true_test_fmri": wandb.Image(rsatoolbox.vis.show_rdm(rdms_true_test_fmri)[0], caption="RDM True Test fMRI")})
    # wandb.log({"rdm_predicted_test_fmri": wandb.Image(rsatoolbox.vis.show_rdm(rdms_predicted_test_fmri)[0], caption="RDM Predicted Test fMRI")})

def main():
    args = parse_args_and_setup_wandb()
    # just to get batch_size for the train dataloader correctly:
    train_dataloader, test_dataloader = get_dataloader(args)
    args.train.batch_size = len(train_dataloader.dataset) # set to the full training set size, not always 9485
    args.validation.batch_size = 515 # set to the full test set size
    # get correctly shaped dataloaders with the updated batch sizes:
    train_dataloader, test_dataloader = get_dataloader(args)
    print(f"Length of the TRAIN dataset: {args.train.batch_size}", flush=True)
    train_fmri_dataset, test_fmri_dataset, train_activations_dataset, test_activations_dataset = get_train_test_numpy_datasets(train_dataloader, test_dataloader, args)
    #################### TEST ###############################
    # train_fmri_dataset, _, train_activations_dataset, _ = get_train_test_numpy_datasets(train_dataloader, test_dataloader, args)
    # args.data.subj = "subj04"
    # train_dataloader, test_dataloader = get_dataloader(args)
    # args.train.batch_size = len(test_dataloader.dataset) # set to the full test set size
    # print("Length of TEST dataset: ", args.train.batch_size, flush=True)
    # train_dataloader, test_dataloader = get_dataloader(args)
    # _, test_fmri_dataset, _, test_activations_dataset = get_train_test_numpy_datasets(train_dataloader, test_dataloader, args)
    #########################################################
    # print("max of train fmri indexes:", train_fmri_dataset.indices.max(), flush=True)
    # print("max of test fmri indexes:", test_fmri_dataset.indices.max(), flush=True)
    # print("max of train activations indexes:", train_activations_dataset.indices.max(), flush=True)
    # print("max of test activations indexes:", test_activations_dataset.indices.max(), flush=True)
    # print()
    # print("min of train fmri indexes:", train_fmri_dataset.indices.min(), flush=True)
    # print("min of test fmri indexes:", test_fmri_dataset.indices.min(), flush=True)
    # print("min of train activations indexes:", train_activations_dataset.indices.min(), flush=True)
    # print("min of test activations indexes:", test_activations_dataset.indices.min(), flush=True)

    # print("Shape of train fmri dataset:", train_fmri_dataset.shape, flush=True)
    # print("Shape of train activations dataset:", train_activations_dataset.shape, flush=True)

    clf = train(train_activations_dataset, train_fmri_dataset, args)    

    #print("Shape of test fmri dataset:", test_fmri_dataset.shape, flush=True)
    #print("Shape of test activations dataset:", test_activations_dataset.shape, flush=True)
    validate_and_visualise(clf, test_activations_dataset, test_fmri_dataset, args)

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cpu") # sklearn works only on CPU
    main()   