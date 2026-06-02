"""Compute fMRI noise ceiling from repeated presentations in NSD.

Reports both lower and upper bounds of the noise ceiling:

  Lower bound (leave-one-out):
    Prediction target excludes the trial being evaluated.

  Upper bound (include-self):
    Prediction target includes the trial being evaluated.  Slightly
    optimistic because the evaluation target shares noise with the
    prediction.

Each bound is computed in two variants:

  Unaveraged:  correlate individual trials pairwise (lower) or a single
    trial vs the mean of all trials (upper).

  Averaged:  correlate each trial with the mean of the *other* trials
    (lower) or the mean of all trials (upper).  Computed empirically,
    not via Spearman-Brown correction.

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

    Two variants for each bound:

      Unaveraged: correlate individual trials pairwise (lower: leave-one-out
        pairs; upper: trial vs mean-of-all-trials).

      Averaged: correlate each trial with the mean of the *other* trials
        for that image (lower), or with the mean of *all* trials including
        itself (upper).  Computed empirically, not via Spearman-Brown.

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

    max_reps = max(len(idxs) for idxs in usable.values())

    # ── Unaveraged lower bound: pairwise trial correlations ──
    pair_r_sums = np.zeros(n_voxels, dtype=np.float64)
    pair_count = 0

    for i, j in combinations(range(max_reps), 2):
        img_ids = [nid for nid, idxs in usable.items() if len(idxs) > max(i, j)]
        if len(img_ids) < 10:
            continue
        resp_i = np.array([betas_roi[usable[nid][i]] for nid in img_ids])
        resp_j = np.array([betas_roi[usable[nid][j]] for nid in img_ids])
        r_pair = _pearsonr_columns(resp_i, resp_j)
        pair_r_sums += r_pair
        pair_count += 1
        print(f"  Unaveraged-lower pair ({i},{j}): {len(img_ids)} images, mean r = {r_pair.mean():.4f}")

    if pair_count == 0:
        raise ValueError("No valid repetition pairs found")

    nc_lower_unavg = pair_r_sums / pair_count
    nc_lower_unavg_clamped = np.clip(nc_lower_unavg, 0, 1)

    # ── Unaveraged upper bound: trial vs mean of ALL trials (include-self) ──
    upper_unavg_r_sums = np.zeros(n_voxels, dtype=np.float64)
    upper_unavg_count = 0

    for trial_idx in range(max_reps):
        img_ids = [nid for nid, idxs in usable.items() if len(idxs) > trial_idx]
        if len(img_ids) < 10:
            continue
        single = np.array([betas_roi[usable[nid][trial_idx]] for nid in img_ids])
        mean_all = np.array([
            np.mean([betas_roi[idx] for idx in usable[nid]], axis=0)
            for nid in img_ids
        ])
        r_upper = _pearsonr_columns(single, mean_all)
        upper_unavg_r_sums += r_upper
        upper_unavg_count += 1
        print(f"  Unaveraged-upper trial {trial_idx}: {len(img_ids)} images, mean r = {r_upper.mean():.4f}")

    nc_upper_unavg = upper_unavg_r_sums / upper_unavg_count
    nc_upper_unavg_clamped = np.clip(nc_upper_unavg, 0, 1)

    # ── Averaged lower bound: trial vs mean of OTHER trials (leave-one-out) ──
    lower_avg_r_sums = np.zeros(n_voxels, dtype=np.float64)
    lower_avg_count = 0

    for trial_idx in range(max_reps):
        img_ids = [nid for nid, idxs in usable.items() if len(idxs) > trial_idx]
        if len(img_ids) < 10:
            continue
        single = np.array([betas_roi[usable[nid][trial_idx]] for nid in img_ids])
        mean_others = np.array([
            np.mean([betas_roi[idx] for idx in usable[nid] if idx != usable[nid][trial_idx]], axis=0)
            for nid in img_ids
        ])
        r_lower = _pearsonr_columns(single, mean_others)
        lower_avg_r_sums += r_lower
        lower_avg_count += 1
        print(f"  Averaged-lower trial {trial_idx}: {len(img_ids)} images, mean r = {r_lower.mean():.4f}")

    nc_lower_avg = lower_avg_r_sums / lower_avg_count
    nc_lower_avg_clamped = np.clip(nc_lower_avg, 0, 1)

    # ── Averaged upper bound: trial vs mean of ALL trials (include-self) ──
    # Same as unaveraged upper — correlating with the averaged signal
    nc_upper_avg = nc_upper_unavg.copy()
    nc_upper_avg_clamped = nc_upper_unavg_clamped.copy()

    return {
        # Lower bound (leave-one-out)
        "nc_lower_unavg": nc_lower_unavg,
        "nc_lower_unavg_clamped": nc_lower_unavg_clamped,
        "nc_lower_avg": nc_lower_avg,
        "nc_lower_avg_clamped": nc_lower_avg_clamped,
        "mean_nc_lower_unavg": float(nc_lower_unavg_clamped.mean()),
        "mean_nc_lower_avg": float(nc_lower_avg_clamped.mean()),
        # Upper bound (include-self)
        "nc_upper_unavg": nc_upper_unavg,
        "nc_upper_unavg_clamped": nc_upper_unavg_clamped,
        "nc_upper_avg": nc_upper_avg,
        "nc_upper_avg_clamped": nc_upper_avg_clamped,
        "mean_nc_upper_unavg": float(nc_upper_unavg_clamped.mean()),
        "mean_nc_upper_avg": float(nc_upper_avg_clamped.mean()),
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

    all_split_results = {}
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
        all_split_results[split_name] = results

        print(f"\n  === LOWER BOUND (leave-one-out) ===")
        print(f"  Unaveraged (pairwise):    {results['mean_nc_lower_unavg']:.4f}")
        print(f"  Averaged (vs mean others):{results['mean_nc_lower_avg']:.4f}")

        print(f"\n  === UPPER BOUND (include-self) ===")
        print(f"  Unaveraged (vs mean all): {results['mean_nc_upper_unavg']:.4f}")
        print(f"  Averaged (vs mean all):   {results['mean_nc_upper_avg']:.4f}")

        # ── Log brain maps to wandb ──
        wandb_payload = {}

        # Scalar summaries
        wandb_payload[f"{split_name}/lower_nc_unaveraged"] = results["mean_nc_lower_unavg"]
        wandb_payload[f"{split_name}/lower_nc_averaged"] = results["mean_nc_lower_avg"]
        wandb_payload[f"{split_name}/upper_nc_unaveraged"] = results["mean_nc_upper_unavg"]
        wandb_payload[f"{split_name}/upper_nc_averaged"] = results["mean_nc_upper_avg"]

        # Brain maps — lower bound
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_lower_unavg_clamped"], roi_indices,
                title=f"NC lower unaveraged ({split_name})",
                wandb_key=f"{split_name}/lower_nc_unaveraged",
            )
        )
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_lower_avg_clamped"], roi_indices,
                title=f"NC lower averaged ({split_name})",
                wandb_key=f"{split_name}/lower_nc_averaged",
            )
        )

        # Brain maps — upper bound
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_upper_unavg_clamped"], roi_indices,
                title=f"NC upper unaveraged ({split_name})",
                wandb_key=f"{split_name}/upper_nc_unaveraged",
            )
        )
        wandb_payload.update(
            visualise_noise_ceiling(
                args, results["nc_upper_avg_clamped"], roi_indices,
                title=f"NC upper averaged ({split_name})",
                wandb_key=f"{split_name}/upper_nc_averaged",
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

    # Save test-split results (the partition used for model evaluation)
    test_results = all_split_results["test"]
    np.savez(
        out_path,
        # Lower bound
        nc_lower_unavg=test_results["nc_lower_unavg"],
        nc_lower_unavg_clamped=test_results["nc_lower_unavg_clamped"],
        nc_lower_avg=test_results["nc_lower_avg"],
        nc_lower_avg_clamped=test_results["nc_lower_avg_clamped"],
        # Upper bound
        nc_upper_unavg=test_results["nc_upper_unavg"],
        nc_upper_unavg_clamped=test_results["nc_upper_unavg_clamped"],
        nc_upper_avg=test_results["nc_upper_avg"],
        nc_upper_avg_clamped=test_results["nc_upper_avg_clamped"],
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
