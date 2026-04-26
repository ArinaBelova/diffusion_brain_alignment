"""Compute inter-subject fMRI noise ceiling from NSD.

For the 515 images shared across all 8 NSD subjects, computes how well
one subject's brain response can be predicted from other subjects' responses.

Reports both lower and upper bounds at two levels:

  **Single-subject (raw):**
    Lower bound (leave-one-out):
      For each subject, correlate their response with the mean of the *other*
      subjects' responses across images.  Average over subjects.
    Upper bound (include-self):
      For each subject, correlate their response with the mean of *all*
      subjects' responses (including themselves).  Average over subjects.

  **Averaged (group of ``group_size`` subjects):**
    For every C(N, group_size) combination of subjects, average their
    responses per image, then compute the same lower/upper bounds
    against the remaining subjects.  This is the empirical analogue of
    Spearman-Brown correction — computed directly instead of projected.

Both levels are reported for:
  - Averaged data   (using trial-averaged betas per subject)
  - Unaveraged data (using per-trial betas, averaged within-subject first
                      to isolate inter-subject variance)
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt

# ── project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion_brain.utils.setup import (
    load_config_from_yaml,
    parse_value,
    set_nested_attr,
)
from diffusion_brain.utils.fmri_behav_data_utils import get_roi_mask, signal_to_2d
from diffusion_brain.utils.visualise import pyplot_brain

ALL_SUBJECTS = [f"subj0{i}" for i in range(1, 9)]


def _pearsonr_columns(a, b):
    """Pearson r between matching columns of a and b.  (n_images, n_voxels) → (n_voxels,)"""
    n = a.shape[0]
    da = a - a.mean(axis=0)
    db = b - b.mean(axis=0)
    cov = (da * db).sum(axis=0) / (n - 1)
    std_a = da.std(axis=0, ddof=1)
    std_b = db.std(axis=0, ddof=1)
    denom = std_a * std_b
    denom[denom == 0] = 1.0
    return cov / denom


def load_averaged_betas_515(subj, roi_indices, behav_data_root, averaged_root):
    """Load trial-averaged betas for the 515 common images, ROI-masked.

    Returns (515, n_roi_voxels) aligned by sorted NSD ID order.
    """
    # NSD IDs for this subject's full averaged dataset
    cond_path = os.path.join(behav_data_root, f"{subj}_all_conditions.npy")
    all_conds = np.load(cond_path, allow_pickle=True)

    # The 515 common NSD IDs
    common_515 = np.load(
        os.path.join(behav_data_root, "common_515_indices.npy"),
        allow_pickle=True,
    )

    # Find positions of the 515 images in this subject's betas
    # Sort by NSD ID so all subjects are aligned identically
    sorted_common = np.sort(common_515)
    pos_indices = np.array([np.where(all_conds == nid)[0][0] for nid in sorted_common])

    # Load full betas (voxels, samples) and extract ROI + 515 images
    betas_path = os.path.join(averaged_root, f"{subj}_betas_average_fsaverage.npy")
    full_betas = np.load(betas_path, mmap_mode="r")  # (327684, n_samples)
    betas_roi = full_betas[roi_indices, :][:, pos_indices].T  # (515, n_roi_voxels)
    return np.array(betas_roi)


def load_unaveraged_betas_515(subj, roi_indices, behav_data_root, unaveraged_root):
    """Load unaveraged betas, average within-subject per image, for the 515 common images.

    Returns (515, n_roi_voxels) aligned by sorted NSD ID order.
    """
    nsd_ids = np.load(os.path.join(unaveraged_root, subj, "nsd_ids.npy"))
    full_betas = np.load(
        os.path.join(unaveraged_root, subj, "betas_fsaverage_unaveraged.npy"),
        mmap_mode="r",
    )  # (n_trials, 327684)

    common_515 = np.load(
        os.path.join(behav_data_root, "common_515_indices.npy"),
        allow_pickle=True,
    )
    sorted_common = np.sort(common_515)
    common_set = set(sorted_common.tolist())

    # Group trial indices by NSD ID (only for the 515 images)
    groups = {}
    for idx, nid in enumerate(nsd_ids):
        nid_int = int(nid)
        if nid_int in common_set:
            groups.setdefault(nid_int, []).append(idx)

    # Average trials within subject for each image, ROI-masked
    n_voxels = len(roi_indices)
    result = np.zeros((len(sorted_common), n_voxels), dtype=np.float32)
    for i, nid in enumerate(sorted_common):
        trial_idxs = groups.get(int(nid), [])
        if len(trial_idxs) == 0:
            continue
        trials = np.array([full_betas[idx, roi_indices] for idx in trial_idxs])
        result[i] = trials.mean(axis=0)

    return result


def compute_intersubject_nc(subject_betas):
    """Compute inter-subject noise ceiling (lower & upper bounds).

    Parameters
    ----------
    subject_betas : dict  {subj_name: ndarray (n_images, n_voxels)}
        Response matrices aligned by image order.

    Returns
    -------
    dict with per-voxel arrays and scalar summaries.
    """
    subj_names = sorted(subject_betas.keys())
    n_subj = len(subj_names)
    n_voxels = next(iter(subject_betas.values())).shape[1]

    # Stack all subjects: (n_subj, n_images, n_voxels)
    stack = np.array([subject_betas[s] for s in subj_names])

    # ── Pairwise correlations (full matrix + mean summary) ──
    # confusion_matrix[i, j] = mean over voxels of corr(subj_i, subj_j)
    confusion_matrix = np.zeros((n_subj, n_subj), dtype=np.float64)
    pair_r_sums = np.zeros(n_voxels, dtype=np.float64)
    pair_count = 0
    for i, j in combinations(range(n_subj), 2):
        r_pair = _pearsonr_columns(stack[i], stack[j])
        mean_r = float(r_pair.mean())
        confusion_matrix[i, j] = mean_r
        confusion_matrix[j, i] = mean_r
        pair_r_sums += r_pair
        pair_count += 1
        print(f"  Pair ({subj_names[i]}, {subj_names[j]}): mean r = {mean_r:.4f}")

    # Diagonal: within-subject (self-correlation = 1.0 by definition)
    np.fill_diagonal(confusion_matrix, 1.0)

    r_pairwise = pair_r_sums / pair_count

    # ── Lower bound: leave-one-subject-out ──
    lower_r_sums = np.zeros(n_voxels, dtype=np.float64)
    for i in range(n_subj):
        # Mean of all OTHER subjects
        others = np.delete(stack, i, axis=0).mean(axis=0)  # (n_images, n_voxels)
        r_loo = _pearsonr_columns(stack[i], others)
        lower_r_sums += r_loo
        print(f"  Lower-bound {subj_names[i]} vs others: mean r = {r_loo.mean():.4f}")
    r_lower = lower_r_sums / n_subj

    # ── Upper bound: include-self ──
    upper_r_sums = np.zeros(n_voxels, dtype=np.float64)
    mean_all = stack.mean(axis=0)  # (n_images, n_voxels)
    for i in range(n_subj):
        r_inc = _pearsonr_columns(stack[i], mean_all)
        upper_r_sums += r_inc
        print(f"  Upper-bound {subj_names[i]} vs all: mean r = {r_inc.mean():.4f}")
    r_upper = upper_r_sums / n_subj

    # Clamp
    r_pairwise_c = np.clip(r_pairwise, 0, 1)
    r_lower_c = np.clip(r_lower, 0, 1)
    r_upper_c = np.clip(r_upper, 0, 1)

    return {
        "r_pairwise": r_pairwise,
        "r_pairwise_clamped": r_pairwise_c,
        "r_lower": r_lower,
        "r_lower_clamped": r_lower_c,
        "r_upper": r_upper,
        "r_upper_clamped": r_upper_c,
        "confusion_matrix": confusion_matrix,
        "mean_r_pairwise": float(r_pairwise_c.mean()),
        "mean_r_lower": float(r_lower_c.mean()),
        "mean_r_upper": float(r_upper_c.mean()),
        "n_subjects": n_subj,
        "subjects": subj_names,
    }


def visualise_nc(args, nc_per_voxel, title, wandb_key):
    """Log noise ceiling brain maps to wandb (2D flatmap + pycortex surface)."""
    payload = {}

    nc_2d, _ = signal_to_2d(args, one_signal_to_transform=nc_per_voxel)
    nc_2d_img = np.squeeze(nc_2d)

    # Use a sequential colormap (0 to max) since NC is non-negative
    from matplotlib.colors import TwoSlopeNorm
    img = nc_2d_img.copy()
    abs_max = max(abs(img.min()), abs(img.max()), 1e-8)
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap='RdBu_r', origin="lower",
              norm=TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max))
    plt.colorbar(im, ax=ax)
    payload[f"{wandb_key}_2d"] = wandb.Image(fig)
    plt.close(fig)
    
    vmax = max(nc_2d_img.max(), 0.01)
    fig_brain = pyplot_brain(nc_per_voxel, args=args,
                             savename=wandb_key, figpath="/tmp",
                             max_cmap_val=vmax)
    payload[f"{wandb_key}_brain"] = wandb.Image(fig_brain)
    plt.close(fig_brain)

    return payload


def plot_pairwise_confusion_matrix(confusion_matrix, subj_names, title):
    """Plot pairwise inter-subject correlation as a confusion matrix heatmap."""
    n = len(subj_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion_matrix, cmap="RdBu_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Mean Pearson r")

    # Tick labels
    short_names = [s.replace("subj0", "S") for s in subj_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            color = "white" if confusion_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{confusion_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=8, color=color)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute inter-subject fMRI noise ceiling")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subjects to include (default: all 8)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--jobid", type=str, default=None)
    args_cli = parser.parse_args()

    args = load_config_from_yaml(args_cli.config)
    for override in args_cli.override:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        set_nested_attr(args, key, parse_value(value))
        print(f"Override: {key} = {parse_value(value)}")

    subjects = args_cli.subjects or ALL_SUBJECTS
    roi_label = f"{args.data.roi_file}_{args.data.roi}"

    run_id = args_cli.jobid or f"intersubj-nc-{roi_label}"
    wandb.init(
        id=run_id,
        name=run_id,
        project="noise-ceiling-fmri",
        config={
            "subjects": subjects,
            "roi": args.data.roi,
            "roi_file": args.data.roi_file,
            "type": "inter-subject",
        },
    )

    print(f"\n{'='*60}")
    print(f"Inter-subject noise ceiling")
    print(f"Subjects: {subjects}")
    print(f"ROI: {args.data.roi_file} / {args.data.roi}")
    print(f"{'='*60}\n")

    roi_indices = get_roi_mask(args)
    print(f"ROI mask: {len(roi_indices)} voxels")

    behav_root = args.data.behav_data_root
    averaged_root = "/data/datapool3/datasets/nsd_betas_condavg/"
    unaveraged_root = "/data/datapool3/datasets/full_nsd_betas/processed/"

    # ── Averaged betas (515 common images) ──
    print(f"\n{'─'*50}")
    print("Loading AVERAGED betas (515 common images)...")
    print(f"{'─'*50}")
    avg_betas = {}
    for subj in subjects:
        print(f"  Loading {subj}...", end=" ", flush=True)
        avg_betas[subj] = load_averaged_betas_515(
            subj, roi_indices, behav_root, averaged_root,
        )
        print(f"shape: {avg_betas[subj].shape}")

    print("\nComputing inter-subject NC on averaged betas:")
    avg_results = compute_intersubject_nc(avg_betas)

    print(f"\n  === AVERAGED DATA (515 images, k=3 within-subject) ===")
    print(f"  Mean pairwise r:      {avg_results['mean_r_pairwise']:.4f}")
    print(f"  NC lower (leave-one-out): {avg_results['mean_r_lower']:.4f}")
    print(f"  NC upper (include-self):  {avg_results['mean_r_upper']:.4f}")

    # ── Unaveraged betas (515 common images, within-subject averaged) ──
    print(f"\n{'─'*50}")
    print("Loading UNAVERAGED betas (515 common images, averaging within-subject)...")
    print(f"{'─'*50}")
    unavg_betas = {}
    for subj in subjects:
        print(f"  Loading {subj}...", end=" ", flush=True)
        unavg_betas[subj] = load_unaveraged_betas_515(
            subj, roi_indices, behav_root, unaveraged_root,
        )
        print(f"shape: {unavg_betas[subj].shape}")

    print("\nComputing inter-subject NC on within-subject-averaged unaveraged betas:")
    unavg_results = compute_intersubject_nc(unavg_betas)

    print(f"\n  === UNAVERAGED DATA (515 images, within-subject averaged) ===")
    print(f"  Mean pairwise r:      {unavg_results['mean_r_pairwise']:.4f}")
    print(f"  NC lower (leave-one-out): {unavg_results['mean_r_lower']:.4f}")
    print(f"  NC upper (include-self):  {unavg_results['mean_r_upper']:.4f}")

    # ── Log to wandb ──
    wandb_payload = {}

    for label, results in [("averaged", avg_results), ("unaveraged", unavg_results)]:
        wandb_payload[f"{label}/mean_r_pairwise"] = results["mean_r_pairwise"]
        wandb_payload[f"{label}/lower_nc"] = results["mean_r_lower"]
        wandb_payload[f"{label}/upper_nc"] = results["mean_r_upper"]

        wandb_payload.update(visualise_nc(
            args, results["r_lower_clamped"],
            title=f"Inter-subj NC lower ({label})",
            wandb_key=f"{label}/lower_nc",
        ))
        wandb_payload.update(visualise_nc(
            args, results["r_upper_clamped"],
            title=f"Inter-subj NC upper ({label})",
            wandb_key=f"{label}/upper_nc",
        ))
        wandb_payload.update(visualise_nc(
            args, results["r_pairwise_clamped"],
            title=f"Inter-subj pairwise r ({label})",
            wandb_key=f"{label}/r_pairwise",
        ))

        # Pairwise confusion matrix
        fig_cm = plot_pairwise_confusion_matrix(
            results["confusion_matrix"],
            results["subjects"],
            title=f"Inter-subject pairwise r ({label})",
        )
        wandb_payload[f"{label}/pairwise_confusion_matrix"] = wandb.Image(fig_cm)
        plt.close(fig_cm)

    wandb.log(wandb_payload)

    # ── Save ──
    if args_cli.output:
        out_path = args_cli.output
    else:
        out_dir = os.path.join("results", "noise_ceiling")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"noise_ceiling_intersubject_{args.data.roi_file}_{args.data.roi}.npz",
        )

    np.savez(
        out_path,
        # Averaged
        avg_r_pairwise=avg_results["r_pairwise_clamped"],
        avg_r_lower=avg_results["r_lower_clamped"],
        avg_r_upper=avg_results["r_upper_clamped"],
        avg_confusion_matrix=avg_results["confusion_matrix"],
        # Unaveraged (within-subject averaged)
        unavg_r_pairwise=unavg_results["r_pairwise_clamped"],
        unavg_r_lower=unavg_results["r_lower_clamped"],
        unavg_r_upper=unavg_results["r_upper_clamped"],
        unavg_confusion_matrix=unavg_results["confusion_matrix"],
        # Metadata
        roi_indices=roi_indices,
        subjects=np.array(subjects),
        roi=args.data.roi,
        roi_file=args.data.roi_file,
    )
    print(f"\nResults saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
