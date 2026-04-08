"""Compute fMRI noise ceiling from repeated presentations in NSD.

Reports both lower and upper bounds of the noise ceiling:

  Lower bound (leave-one-out):
    For each trial, correlate it with the mean of the *other* trials across
    images.  This is the performance a perfect model would achieve when
    evaluated against held-out data.

  Upper bound (include-self):
    For each trial, correlate it with the mean of *all* trials (including
    itself) across images.  Slightly optimistic because the evaluation
    target shares noise with the prediction.

Both bounds are reported for:
  - Unaveraged data  (single-trial prediction)
  - Averaged data    (k-trial averaged prediction, Spearman-Brown corrected)

Works on both 1D voxel vectors and 2D flatmap projections.
Reports per-voxel noise ceiling averaged over active voxels.
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── project imports (run inside container with PYTHONPATH set) ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion_brain.utils.setup import (
    load_config_from_yaml,
    parse_value,
    set_nested_attr,
)
from diffusion_brain.utils.fmri_behav_data_utils import get_roi_mask, signal_to_2d
from diffusion_brain.utils.visualise import fmri_to_wandb_image, pyplot_brain


def build_repetition_groups(nsd_ids):
    """Group trial indices by NSD image ID.

    Returns dict  {nsd_id: [row_idx_0, row_idx_1, ...]}
    """
    groups = {}
    for idx, nsd_id in enumerate(nsd_ids):
        groups.setdefault(int(nsd_id), []).append(idx)
    return groups


def _pearsonr_columns(a, b):
    """Pearson r between matching columns of a and b.  (n_images, n_voxels) → (n_voxels,)"""
    n = a.shape[0]
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    da = a - mean_a
    db = b - mean_b
    cov = (da * db).sum(axis=0) / (n - 1)
    std_a = da.std(axis=0, ddof=1)
    std_b = db.std(axis=0, ddof=1)
    denom = std_a * std_b
    denom[denom == 0] = 1.0
    return cov / denom


def compute_noise_ceiling_1d(betas_roi, nsd_ids):
    """Compute per-voxel noise ceiling (lower & upper bounds).

    Parameters
    ----------
    betas_roi : ndarray, shape (n_trials, n_voxels)
        Single-trial fMRI betas for the ROI.
    nsd_ids : ndarray, shape (n_trials,)
        NSD image ID for each trial row.

    Returns
    -------
    dict with per-voxel arrays and scalar summaries for both bounds.
    """
    groups = build_repetition_groups(nsd_ids)

    # Count repetitions
    rep_counts = {}
    for nsd_id, idxs in groups.items():
        k = len(idxs)
        rep_counts[k] = rep_counts.get(k, 0) + 1
    print(f"Repetition distribution: { {k: rep_counts[k] for k in sorted(rep_counts)} }")

    # Use images with ≥ 2 reps
    usable = {nsd_id: idxs for nsd_id, idxs in groups.items() if len(idxs) >= 2}
    n_images = len(usable)
    n_voxels = betas_roi.shape[1]
    print(f"Images with ≥2 repetitions: {n_images}, voxels: {n_voxels}")

    if n_images == 0:
        raise ValueError("No images with ≥2 repetitions found")

    # ── Build per-image trial matrices ──
    # For simplicity, work with images that have exactly the modal rep count.
    # Most NSD images have 3 reps; handle variable counts gracefully.
    max_reps = max(len(idxs) for idxs in usable.values())

    # ── Lower bound: pairwise correlations (leave-one-out) ──
    # Average r across all distinct trial pairs.  This estimates r_single,
    # from which we derive NC_lower via Spearman-Brown.
    pair_r_sums = np.zeros(n_voxels, dtype=np.float64)
    pair_count = 0

    for i, j in combinations(range(max_reps), 2):
        img_ids = [nid for nid, idxs in usable.items() if len(idxs) > max(i, j)]
        if len(img_ids) < 10:
            continue
        resp_i = np.array([betas_roi[usable[nid][i]] for nid in img_ids])
        resp_j = np.array([betas_roi[usable[nid][j]] for nid in img_ids])
        # r_pair = _pearsonr_columns(resp_i, resp_j)
        r_pair = np.array([scipy.stats.pearsonr(resp_i[:, v], resp_j[:, v])[0] for v in range(n_voxels)])
        pair_r_sums += r_pair
        pair_count += 1
        print(f"  Lower-bound pair ({i},{j}): {len(img_ids)} images, mean r = {r_pair.mean():.4f}")

    if pair_count == 0:
        raise ValueError("No valid repetition pairs found")

    r_single = pair_r_sums / pair_count
    r_single_clamped = np.clip(r_single, 0, 1)

    # ── Upper bound: correlate each trial with mean of ALL trials ──
    # r_upper_i = corr(trial_i, mean_all) across images.  Average over trials.
    upper_r_sums = np.zeros(n_voxels, dtype=np.float64)
    upper_count = 0

    for trial_idx in range(max_reps):
        img_ids = [nid for nid, idxs in usable.items() if len(idxs) > trial_idx]
        if len(img_ids) < 10:
            continue

        # Single trial responses: (n_images, n_voxels)
        single = np.array([betas_roi[usable[nid][trial_idx]] for nid in img_ids])

        # Mean of ALL trials for each image (including this trial)
        mean_all = np.array([
            np.mean([betas_roi[idx] for idx in usable[nid]], axis=0)
            for nid in img_ids
        ])

        r_upper = _pearsonr_columns(single, mean_all)
        upper_r_sums += r_upper
        upper_count += 1
        print(f"  Upper-bound trial {trial_idx}: {len(img_ids)} images, mean r = {r_upper.mean():.4f}")

    r_upper_avg = upper_r_sums / upper_count
    r_upper_clamped = np.clip(r_upper_avg, 0, 1)

    # ── Derive noise ceilings ──
    # Lower bound: Spearman-Brown on pairwise r_single
    nc_lower_unavg = r_single_clamped  # NC for predicting a single trial
    nc_lower_avg = {}
    for k in [2, 3]:
        nc_lower_avg[k] = (k * r_single_clamped) / (1 + (k - 1) * r_single_clamped)

    # Upper bound: direct correlation with all-trial mean
    # For averaged prediction, this IS the upper bound directly (the target
    # is the k-trial mean, and we correlated against exactly that).
    # For single-trial prediction, the upper bound uses pairwise r but
    # the include-self version inflates it, so we report r_upper as-is.
    nc_upper_unavg = r_upper_clamped
    nc_upper_avg = {}
    for k in [2, 3]:
        nc_upper_avg[k] = (k * r_upper_clamped) / (1 + (k - 1) * r_upper_clamped)

    return {
        # Lower bound (leave-one-out)
        "r_single": r_single,
        "r_single_clamped": r_single_clamped,
        "nc_lower_unavg": nc_lower_unavg,
        "nc_lower_avg": nc_lower_avg,
        "mean_r_single": float(r_single.mean()),
        "mean_r_single_clamped": float(r_single_clamped.mean()),
        "mean_nc_lower_unavg": float(nc_lower_unavg.mean()),
        "mean_nc_lower_avg": {k: float(v.mean()) for k, v in nc_lower_avg.items()},
        # Upper bound (include-self)
        "r_upper": r_upper_avg,
        "r_upper_clamped": r_upper_clamped,
        "nc_upper_unavg": nc_upper_unavg,
        "nc_upper_avg": nc_upper_avg,
        "mean_r_upper": float(r_upper_avg.mean()),
        "mean_r_upper_clamped": float(r_upper_clamped.mean()),
        "mean_nc_upper_unavg": float(nc_upper_unavg.mean()),
        "mean_nc_upper_avg": {k: float(v.mean()) for k, v in nc_upper_avg.items()},
        # Metadata
        "n_images_used": n_images,
        "rep_counts": rep_counts,
    }


def visualise_noise_ceiling(args, nc_per_voxel, roi_indices, title, wandb_key):
    """Log noise ceiling brain maps to wandb (2D flatmap + pycortex surface)."""
    payload = {}

    # ── 2D flatmap projection ──
    nc_2d, locations = signal_to_2d(args, one_signal_to_transform=nc_per_voxel)
    nc_2d_img = np.squeeze(nc_2d)  # (H, W)

    # Use a sequential colormap (0 to max) since NC is non-negative
    from matplotlib.colors import TwoSlopeNorm
    img = nc_2d_img.copy()
    abs_max = max(abs(img.min()), abs(img.max()), 1e-8)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='RdBu_r', origin="lower",
              norm=TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max))
    if title:
        ax.set_title(title)
    plt.colorbar(ax.images[0], ax=ax)
    payload[f"{wandb_key}_2d"] = wandb.Image(fig)
    plt.close(fig)

    # fig, ax = plt.subplots()
    vmax = max(nc_2d_img.max(), 0.01)
    # im = ax.imshow(nc_2d_img, cmap="RdBu_r", origin="lower", vmin=0, vmax=vmax)
    # ax.set_title(title)
    # plt.colorbar(im, ax=ax)
    # payload[f"{wandb_key}_2d"] = wandb.Image(fig)
    # plt.close(fig)

    # ── Pycortex surface view ──
    fig_brain = pyplot_brain(nc_per_voxel, args=args,
                             savename=wandb_key, figpath="/tmp",
                             max_cmap_val=vmax)
    payload[f"{wandb_key}_brain"] = wandb.Image(fig_brain)
    plt.close(fig_brain)

    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute fMRI noise ceiling")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (uses data section for ROI/paths)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides as key=value")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results (.npz)")
    parser.add_argument("--jobid", type=str, default=None,
                        help="Job ID for wandb run name")
    args_cli = parser.parse_args()

    # Load config (reuse the same loader as train.py)
    args = load_config_from_yaml(args_cli.config)
    for override in args_cli.override:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        set_nested_attr(args, key, parse_value(value))
        print(f"Override: {key} = {parse_value(value)}")

    subj = args.data.subj
    roi_label = f"{args.data.roi_file}_{args.data.roi}"

    # ── Init wandb ──
    run_id = args_cli.jobid or f"noise-ceiling-{subj}-{roi_label}"
    wandb.init(
        id=run_id,
        name=run_id,
        project="noise-ceiling-fmri",
        config={
            "subj": subj,
            "roi": args.data.roi,
            "roi_file": args.data.roi_file,
        },
    )

    print(f"\n{'='*60}")
    print(f"Noise ceiling computation for {subj}")
    print(f"ROI: {args.data.roi_file} / {args.data.roi}")
    print(f"{'='*60}\n")

    # ── Load unaveraged single-trial betas ──
    # We always need the unaveraged betas for noise ceiling, regardless of
    # which dataset variant the config specifies for training.
    unaveraged_root = "/data/datapool3/datasets/full_nsd_betas/processed/"
    nsd_ids_path = os.path.join(unaveraged_root, subj, "nsd_ids.npy")
    betas_path = os.path.join(unaveraged_root, subj, "betas_fsaverage_unaveraged.npy")

    print(f"Loading NSD IDs from {nsd_ids_path}")
    nsd_ids = np.load(nsd_ids_path)
    print(f"  shape: {nsd_ids.shape}, unique images: {len(np.unique(nsd_ids))}")

    print(f"Loading unaveraged betas from {betas_path} (memory-mapped)")
    full_betas = np.load(betas_path, mmap_mode="r")  # (n_trials, 327684)
    print(f"  shape: {full_betas.shape}")

    # ── Extract ROI voxels ──
    roi_indices = get_roi_mask(args)
    print(f"ROI mask: {len(roi_indices)} voxels")

    betas_roi = full_betas[:, roi_indices]  # (n_trials, n_roi_voxels)
    # Force into memory (mmap slicing can be slow for column access)
    print("Loading ROI betas into memory...")
    betas_roi = np.array(betas_roi)
    print(f"  ROI betas shape: {betas_roi.shape}")

    # ── Compute on train/test splits separately ──
    test_nsd_ids_set = set(
        np.load(
            os.path.join(args.data.behav_data_root, "common_515_indices.npy"),
            allow_pickle=True,
        ).tolist()
    )

    for split_name, split_mask in [
        ("test", np.array([int(nid) in test_nsd_ids_set for nid in nsd_ids])),
        ("train", np.array([int(nid) not in test_nsd_ids_set for nid in nsd_ids])),
        ("all", np.ones(len(nsd_ids), dtype=bool)),
    ]:
        print(f"\n{'─'*50}")
        print(f"Split: {split_name}  ({split_mask.sum()} trials)")
        print(f"{'─'*50}")

        split_betas = betas_roi[split_mask]
        split_nsd_ids = nsd_ids[split_mask]

        results = compute_noise_ceiling_1d(split_betas, split_nsd_ids)

        print(f"\n  === LOWER BOUND (leave-one-out) ===")
        print(f"  Single-trial reliability: {results['mean_r_single_clamped']:.4f}")
        print(f"  NC unaveraged:            {results['mean_nc_lower_unavg']:.4f}")
        for k, nc in sorted(results["mean_nc_lower_avg"].items()):
            print(f"  NC averaged k={k}:         {nc:.4f}")

        print(f"\n  === UPPER BOUND (include-self) ===")
        print(f"  Single-trial reliability: {results['mean_r_upper_clamped']:.4f}")
        print(f"  NC unaveraged:            {results['mean_nc_upper_unavg']:.4f}")
        for k, nc in sorted(results["mean_nc_upper_avg"].items()):
            print(f"  NC averaged k={k}:         {nc:.4f}")

        # ── Log brain maps to wandb ──
        wandb_payload = {}

        # Scalar summaries
        wandb_payload[f"{split_name}/lower_r_single"] = results["mean_r_single_clamped"]
        wandb_payload[f"{split_name}/lower_nc_unaveraged"] = results["mean_nc_lower_unavg"]
        for k, nc in results["mean_nc_lower_avg"].items():
            wandb_payload[f"{split_name}/lower_nc_averaged_k{k}"] = nc

        wandb_payload[f"{split_name}/upper_r_single"] = results["mean_r_upper_clamped"]
        wandb_payload[f"{split_name}/upper_nc_unaveraged"] = results["mean_nc_upper_unavg"]
        for k, nc in results["mean_nc_upper_avg"].items():
            wandb_payload[f"{split_name}/upper_nc_averaged_k{k}"] = nc

        # Brain maps — lower bound
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_lower_unavg"], roi_indices,
                title=f"NC lower unaveraged ({split_name})",
                wandb_key=f"{split_name}/lower_nc_unaveraged",
            )
        )
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_lower_avg"][3], roi_indices,
                title=f"NC lower averaged k=3 ({split_name})",
                wandb_key=f"{split_name}/lower_nc_averaged_k3",
            )
        )

        # Brain maps — upper bound
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_upper_unavg"], roi_indices,
                title=f"NC upper unaveraged ({split_name})",
                wandb_key=f"{split_name}/upper_nc_unaveraged",
            )
        )
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_upper_avg"][3], roi_indices,
                title=f"NC upper averaged k=3 ({split_name})",
                wandb_key=f"{split_name}/upper_nc_averaged_k3",
            )
        )

        wandb.log(wandb_payload)

    # ── Save results ──
    if args_cli.output:
        out_path = args_cli.output
    else:
        out_dir = os.path.join("results", "noise_ceiling")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"noise_ceiling_{subj}_{args.data.roi_file}_{args.data.roi}.npz",
        )

    # Re-compute on all data for saving
    all_results = compute_noise_ceiling_1d(betas_roi, nsd_ids)
    np.savez(
        out_path,
        # Lower bound
        r_single=all_results["r_single"],
        r_single_clamped=all_results["r_single_clamped"],
        nc_lower_unavg=all_results["nc_lower_unavg"],
        **{f"nc_lower_avg_k{k}": v for k, v in all_results["nc_lower_avg"].items()},
        # Upper bound
        r_upper=all_results["r_upper"],
        r_upper_clamped=all_results["r_upper_clamped"],
        nc_upper_unavg=all_results["nc_upper_unavg"],
        **{f"nc_upper_avg_k{k}": v for k, v in all_results["nc_upper_avg"].items()},
        # Metadata
        roi_indices=roi_indices,
        subj=subj,
        roi=args.data.roi,
        roi_file=args.data.roi_file,
    )
    print(f"\nResults saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
