"""Cross-subject confusion matrix for the group RidgeCV baseline.

For each subject: trains voxel-wise RidgeCV (ANN activations -> fMRI) on
the training set.  Then on the shared 515 test stimuli, computes an
N_subj x N_subj confusion matrix where entry (i, j) = mean per-voxel
Pearson r between subject i's Ridge predictions and subject j's true fMRI.

The diagonal should match the per-subject r from train_sklearn_group.py.
The off-diagonal measures cross-subject transferability of the linear
model; specificity = mean(diag) - mean(off) quantifies whether
per-subject Ridge models produce subject-specific predictions.

Mirrors the diffusion model's _evaluate_cross_subject_confusion in
generate.py so the two can be compared directly.
"""

from sklearn.linear_model import RidgeCV
import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import wandb
import os

from diffusion_brain.datasets.paired_brain_ann_dataset import PairedBrainAnnDataset
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.fmri_behav_data_utils import ALL_SUBJECTS, get_train_test_indices, ensure_fmri_roi_exists
from diffusion_brain.utils.ann_activations_utils import ensure_activations_exist


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


def per_voxel_r(pred, true):
    """Per-voxel Pearson r between pred and true (n_images, n_voxels)."""
    n_voxels = true.shape[1]
    r = np.full(n_voxels, np.nan, dtype=np.float64)
    for v in range(n_voxels):
        p = pred[:, v]
        t = true[:, v]
        if np.std(p) == 0 or np.std(t) == 0:
            continue
        r[v] = pearsonr(t, p)[0]
    return r


def main():
    args = parse_args_and_setup_wandb()
    subjects = getattr(args.data, "subjects", ALL_SUBJECTS)
    is_2d = args.data.is_2d

    # Phase 1: Train per-subject Ridge models and collect test predictions/true fMRI
    per_subject_models = {}   # subj -> fitted RidgeCV
    per_subject_test_pred = {}  # subj -> (n_stim, n_voxels) predictions
    per_subject_test_true = {}  # subj -> (n_stim, n_voxels) true fMRI
    locations_roi = None
    image_shape = None

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"Training Ridge for {subj}")
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

        # Fit RidgeCV
        alphas = np.array([1e-3, 1e-2, 1e-1, 1.0])
        clf = RidgeCV(alphas=alphas, alpha_per_target=True)
        clf.fit(train_ann, train_fmri_1d)

        pred = clf.predict(test_ann)

        per_subject_models[subj] = clf
        per_subject_test_pred[subj] = pred         # (515, n_voxels)
        per_subject_test_true[subj] = test_fmri_1d  # (515, n_voxels)

        # Quick sanity: per-subject self r
        r_self = per_voxel_r(pred, test_fmri_1d)
        print(f"  {subj}: self mean r = {np.nanmean(r_self):.4f}, "
              f"median r = {np.nanmedian(r_self):.4f}", flush=True)

    # Phase 2: Build confusion matrix
    n_subj = len(subjects)
    subj_names = list(subjects)

    # All subjects share the 515 test stimuli in the same order (common_515),
    # so predictions and true fMRI are already stimulus-aligned.
    n_stim = per_subject_test_pred[subj_names[0]].shape[0]
    n_vox = per_subject_test_pred[subj_names[0]].shape[1]

    confusion = np.full((n_subj, n_subj), np.nan, dtype=np.float64)
    for i_idx, subj_i in enumerate(subj_names):
        for j_idx, subj_j in enumerate(subj_names):
            # Row i = predictions from subject i's model
            # Col j = true fMRI from subject j
            r_vox = per_voxel_r(per_subject_test_pred[subj_i],
                                per_subject_test_true[subj_j])
            confusion[i_idx, j_idx] = float(np.nanmean(r_vox))

    # Print the matrix
    header = "Cross-subject Ridge confusion matrix"
    print(f"\n{'=' * 60}\n{header}\n{'=' * 60}", flush=True)
    print(f"  rows = Ridge model trained on subject i", flush=True)
    print(f"  cols = true fMRI from subject j", flush=True)
    print(f"  cell = mean per-voxel Pearson r over {n_stim} shared stimuli, "
          f"{n_vox} voxels", flush=True)

    col_header = "             " + "  ".join(f"{n:>8}" for n in subj_names)
    print(col_header, flush=True)
    for i_idx, i_name in enumerate(subj_names):
        row_vals = "  ".join(f"{confusion[i_idx, j_idx]:+8.4f}" for j_idx in range(n_subj))
        print(f"  {i_name:>10}  {row_vals}", flush=True)

    diag_vals = np.diag(confusion)
    off_mask = ~np.eye(n_subj, dtype=bool)
    mean_diag = float(np.nanmean(diag_vals))
    mean_off = float(np.nanmean(confusion[off_mask]))
    specificity = mean_diag - mean_off
    print(f"\n  mean(diagonal)   = {mean_diag:+.4f}", flush=True)
    print(f"  mean(off-diag)   = {mean_off:+.4f}", flush=True)
    print(
        f"  specificity      = {specificity:+.4f}  "
        f"(larger = per-subject Ridge models produce subject-specific predictions)",
        flush=True,
    )
    print(f"{'=' * 60}\n", flush=True)

    # Log to wandb
    log_payload = {
        "confusion/mean_diagonal": mean_diag,
        "confusion/mean_offdiagonal": mean_off,
        "confusion/specificity": specificity,
    }
    for i_idx, i_name in enumerate(subj_names):
        for j_idx, j_name in enumerate(subj_names):
            log_payload[f"confusion/cell/pred_{i_name}_vs_true_{j_name}"] = float(confusion[i_idx, j_idx])

    # Per-subject diagonal r (same as train_sklearn_group.py per-subject mean r)
    for i_idx, subj in enumerate(subj_names):
        log_payload[f"r_mean/{subj}"] = float(diag_vals[i_idx])

    # Render heatmap
    finite = confusion[np.isfinite(confusion)]
    if finite.size > 0:
        vmax = float(np.nanmax(np.abs(finite)))
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(1.1 * n_subj + 2, 1.1 * n_subj + 1.5))
    im = ax.imshow(confusion, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(n_subj))
    ax.set_yticks(range(n_subj))
    ax.set_xticklabels(subj_names, rotation=45, ha="right")
    ax.set_yticklabels(subj_names)
    ax.set_xlabel("true subject (columns)")
    ax.set_ylabel("Ridge model trained on (rows)")
    ax.set_title(
        f"Cross-subject Ridge confusion\n"
        f"diag={mean_diag:+.4f}, off={mean_off:+.4f}, spec={specificity:+.4f}"
    )
    for i_idx in range(n_subj):
        for j_idx in range(n_subj):
            val = confusion[i_idx, j_idx]
            if not np.isfinite(val):
                continue
            txt_color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(
                j_idx, i_idx, f"{val:+.3f}",
                ha="center", va="center", color=txt_color, fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean per-voxel Pearson r")
    fig.tight_layout()
    log_payload["confusion/matrix_heatmap"] = wandb.Image(fig)
    plt.close(fig)

    wandb.log(log_payload)

    # Save raw results
    output_dir = os.path.join(
        getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
        args.jobid,
    )
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "ridge_confusion_matrix.npz"),
        confusion=confusion,
        subject_names=np.array(subj_names),
        mean_diagonal=mean_diag,
        mean_offdiagonal=mean_off,
        specificity=specificity,
        n_stim=n_stim,
        n_voxels=n_vox,
    )
    print(f"\nResults saved to {output_dir}/ridge_confusion_matrix.npz", flush=True)


if __name__ == "__main__":
    main()
