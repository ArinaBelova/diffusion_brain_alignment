"""Group-average linear encoding model across all NSD subjects.

For each subject: trains voxel-wise RidgeCV (ANN activations -> fMRI),
computes per-voxel Pearson r on the test set.  Then:
  - Group-averages the per-voxel r maps across N=8 subjects
  - Per-voxel two-tailed t-test across participants
  - Benjamini-Hochberg FDR correction at P=0.05
  - Logs the thresholded group-average r map to wandb
"""

from sklearn.linear_model import RidgeCV
import torch
import numpy as np
from scipy.stats import pearsonr, ttest_1samp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import wandb
import os

from diffusion_brain.datasets.paired_brain_ann_dataset import PairedBrainAnnDataset
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.fmri_behav_data_utils import ALL_SUBJECTS, get_train_test_indices, ensure_fmri_roi_exists
from diffusion_brain.utils.ann_activations_utils import ensure_activations_exist

FDR_ALPHA = 0.05


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (reject, p_corrected)."""
    m = len(p_values)
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]

    # BH adjusted p-values
    p_corrected = np.empty(m)
    cummin = 1.0
    for i in range(m - 1, -1, -1):
        adjusted = sorted_p[i] * m / (i + 1)
        cummin = min(cummin, adjusted)
        p_corrected[sort_idx[i]] = min(cummin, 1.0)

    reject = p_corrected <= alpha
    return reject, p_corrected


def load_subject_data(args, subj):
    """Load train/test numpy arrays for a single subject.

    Uses per-subject activation cache paths to avoid the shared-count
    collision in get_ann_brain_dataloader (all 8 subjects have 9485
    train images but different NSD IDs).
    """
    args.data.subj = subj

    train_nsd_ids, test_nsd_ids, train_indices, test_indices = get_train_test_indices(args)
    fmri_roi_path = ensure_fmri_roi_exists(args)

    # Per-subject activation paths (include subj to avoid cache collision)
    train_act_path = os.path.join(
        args.data.ann_activations_data_path, args.data.ann_model,
        f"activations_weights_{args.data.ann_model_weights}_layer_{args.data.layer_name}_{subj}_{len(train_nsd_ids)}_samples.pt",
    )
    # Test activations are the same 515 images for all subjects
    test_act_path = os.path.join(
        args.data.ann_activations_data_path, args.data.ann_model,
        f"activations_weights_{args.data.ann_model_weights}_layer_{args.data.layer_name}_{len(test_nsd_ids)}_samples.pt",
    )
    ensure_activations_exist(train_act_path, train_nsd_ids, args)
    ensure_activations_exist(test_act_path, test_nsd_ids, args)

    fmri_norm_mode = getattr(args.data, "fmri_norm_mode", "active_std")

    train_ds = PairedBrainAnnDataset(
        activations_path=train_act_path,
        fmri_roi_path=fmri_roi_path,
        sample_indices=train_indices,
        is_2d=args.data.is_2d,
        fmri_norm_mode=fmri_norm_mode,
    )
    test_ds = PairedBrainAnnDataset(
        activations_path=test_act_path,
        fmri_roi_path=fmri_roi_path,
        sample_indices=test_indices,
        is_2d=args.data.is_2d,
        act_mean=train_ds.act_mean,
        act_std=train_ds.act_std,
        fmri_scale=getattr(train_ds, "fmri_scale", None),
        fmri_norm_mode=fmri_norm_mode,
    )

    # Load everything in one batch
    train_fmri, train_ann = next(iter(torch.utils.data.DataLoader(
        train_ds, batch_size=len(train_ds), shuffle=False)))
    test_fmri, test_ann = next(iter(torch.utils.data.DataLoader(
        test_ds, batch_size=len(test_ds), shuffle=False)))

    return (train_fmri.numpy(), train_ann.numpy(),
            test_fmri.numpy(), test_ann.numpy())


def extract_roi_voxels(fmri_2d, args):
    """Extract active ROI voxel time series from 2D images.

    Returns:
        voxels: (n_images, n_locations) array
        locations_roi: (2, n_locations) coordinate array
    """
    locations_load_path = os.path.join(
        args.data.roi_defs_dir,
        f"roi_preselected_extended_2d_images_res_{args.data.grid_resolution_2d}",
        f"{args.data.roi_file}",
        f"{args.data.subj}_{args.data.roi}.npz",
    )
    locations_roi = np.load(locations_load_path, allow_pickle=True)["locations"]
    y_coords, x_coords = locations_roi[0], locations_roi[1]

    fmri_squeezed = fmri_2d.squeeze(1)  # (N, H, W)
    voxels = fmri_squeezed[:, y_coords, x_coords]  # (N, n_locations)
    return voxels, locations_roi


def fit_and_evaluate(train_ann, train_fmri, test_ann, test_fmri):
    """Train RidgeCV and return per-voxel Pearson r on the test set."""
    alphas = np.array([1e-3, 1e-2, 1e-1, 1.0])

    clf = RidgeCV(alphas=alphas, alpha_per_target=True)
    clf.fit(train_ann, train_fmri)

    pred = clf.predict(test_ann)

    n_voxels = test_fmri.shape[1]
    r_per_voxel = np.array([
        pearsonr(test_fmri[:, v], pred[:, v])[0]
        for v in range(n_voxels)
    ])
    return r_per_voxel


def main():
    args = parse_args_and_setup_wandb()
    subjects = getattr(args.data, "subjects", ALL_SUBJECTS)
    is_2d = args.data.is_2d

    all_r = []  # list of (n_voxels,) arrays, one per subject
    image_shape = None
    locations_roi = None

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"Processing {subj}")
        print(f"{'='*60}", flush=True)

        train_fmri, train_ann, test_fmri, test_ann = load_subject_data(args, subj)

        if is_2d:
            train_fmri_1d, loc = extract_roi_voxels(train_fmri, args)
            test_fmri_1d, _ = extract_roi_voxels(test_fmri, args)
            if image_shape is None:
                image_shape = train_fmri.squeeze(1).shape[1:]  # (H, W)
                locations_roi = loc
        else:
            train_fmri_1d = train_fmri
            test_fmri_1d = test_fmri

        r_per_voxel = fit_and_evaluate(train_ann, train_fmri_1d, test_ann, test_fmri_1d)
        print(f"{subj}: mean r = {np.nanmean(r_per_voxel):.4f}, "
              f"median r = {np.nanmedian(r_per_voxel):.4f}", flush=True)
        all_r.append(r_per_voxel)

    # Stack: (N_subjects, n_voxels)
    all_r = np.array(all_r)
    n_subjects, n_voxels = all_r.shape
    print(f"\nGroup r-score matrix shape: {all_r.shape}", flush=True)

    # Group-average r map
    group_mean_r = np.nanmean(all_r, axis=0)

    # Per-voxel two-tailed t-test (H0: population mean r = 0)
    t_stats, p_values = ttest_1samp(all_r, popmean=0.0, axis=0, nan_policy="omit")

    # Handle NaN p-values (e.g., zero-variance voxels) — treat as non-significant
    nan_mask = np.isnan(p_values)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum()} voxels have NaN p-values (set to 1.0)", flush=True)
        p_values[nan_mask] = 1.0

    # Benjamini-Hochberg FDR correction
    reject, p_corrected = benjamini_hochberg(p_values, alpha=FDR_ALPHA)

    n_sig = reject.sum()
    print(f"\nSignificant voxels (BH-FDR q<{FDR_ALPHA}): {n_sig}/{n_voxels} "
          f"({100*n_sig/n_voxels:.1f}%)", flush=True)
    print(f"Group mean r (all voxels): {np.nanmean(group_mean_r):.4f}", flush=True)
    print(f"Group mean r (significant only): {np.nanmean(group_mean_r[reject]):.4f}" if n_sig > 0 else "No significant voxels", flush=True)

    # Thresholded map: zero out non-significant voxels
    group_mean_r_thresh = group_mean_r.copy()
    group_mean_r_thresh[~reject] = 0.0

    # Log per-subject mean r
    for i, subj in enumerate(subjects):
        wandb.log({f"r_mean/{subj}": np.nanmean(all_r[i])})

    wandb.log({
        "group/mean_r_all_voxels": np.nanmean(group_mean_r),
        "group/mean_r_significant": np.nanmean(group_mean_r[reject]) if n_sig > 0 else 0.0,
        "group/n_significant_voxels": int(n_sig),
        "group/n_total_voxels": int(n_voxels),
        "group/frac_significant": float(n_sig / n_voxels),
    })

    # Visualisation
    if is_2d and image_shape is not None:
        y_coords, x_coords = locations_roi[0], locations_roi[1]

        # Unthresholded group-average r map
        r_img = np.zeros(image_shape, dtype=np.float32)
        r_img[y_coords, x_coords] = group_mean_r
        abs_max = max(abs(r_img.min()), abs(r_img.max()), 1e-8)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        fig_raw, ax_raw = plt.subplots(figsize=(8, 6))
        im = ax_raw.imshow(r_img, cmap="RdBu_r", origin="lower", norm=norm)
        fig_raw.colorbar(im, ax=ax_raw)
        ax_raw.set_title(f"Group-average Pearson r (N={n_subjects}, all voxels)")
        wandb.log({"group/r_map_unthresholded": wandb.Image(fig_raw)})
        plt.close(fig_raw)

        # Thresholded group-average r map (BH-FDR corrected)
        r_img_thresh = np.zeros(image_shape, dtype=np.float32)
        r_img_thresh[y_coords, x_coords] = group_mean_r_thresh
        abs_max_t = max(abs(r_img_thresh.min()), abs(r_img_thresh.max()), 1e-8)
        norm_t = TwoSlopeNorm(vmin=-abs_max_t, vcenter=0, vmax=abs_max_t)

        fig_thresh, ax_thresh = plt.subplots(figsize=(8, 6))
        im_t = ax_thresh.imshow(r_img_thresh, cmap="RdBu_r", origin="lower", norm=norm_t)
        fig_thresh.colorbar(im_t, ax=ax_thresh)
        ax_thresh.set_title(f"Group-average Pearson r (N={n_subjects}, BH-FDR q<{FDR_ALPHA})")
        wandb.log({"group/r_map_fdr_thresholded": wandb.Image(fig_thresh)})
        plt.close(fig_thresh)

        # Significance map (binary)
        sig_img = np.zeros(image_shape, dtype=np.float32)
        sig_img[y_coords, x_coords] = reject.astype(float)

        fig_sig, ax_sig = plt.subplots(figsize=(8, 6))
        ax_sig.imshow(sig_img, cmap="Greys", origin="lower", vmin=0, vmax=1)
        ax_sig.set_title(f"Significant voxels (BH-FDR q<{FDR_ALPHA})")
        wandb.log({"group/significance_mask": wandb.Image(fig_sig)})
        plt.close(fig_sig)

    # Save raw results
    output_dir = os.path.join(
        getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
        args.jobid,
    )
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "group_ridge_results.npz"),
        all_r=all_r,
        group_mean_r=group_mean_r,
        t_stats=t_stats,
        p_values=p_values,
        p_corrected=p_corrected,
        reject=reject,
        subjects=np.array(subjects),
    )
    print(f"\nResults saved to {output_dir}/group_ridge_results.npz", flush=True)


if __name__ == "__main__":
    main()
