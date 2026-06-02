"""Permutation test for within-subject fMRI noise ceiling lower bound (NC-LB) on 2D data.

Operates on the same 2D flatmap representation the diffusion model is trained
on. The ROI signal is projected with `signal_to_2d` once, then the active 2D
pixels (= valid voxels in the 2D representation) are unrolled to a flat
[n_trials, n_pixels] matrix that the existing 1D NC-LB pipeline can consume.

For each pixel, builds a null distribution of NC-LB (leave-one-out, averaged)
by shuffling the image rows of [n_trials, n_pixels] N=5000 times and re-running
the same NC-LB pipeline used in noise_ceiling.py.

Reuses the existing repeat-extraction (`build_repetition_groups`) and
correlation primitives (`_pearsonr_columns`) from noise_ceiling.py — the
slim `compute_nc_lb_avg` helper below mirrors the `nc_lower_avg` block of
`compute_noise_ceiling_1d` exactly.

Outputs (all shape [n_pixels]):
  nc_lb_real : real NC-LB per active 2D pixel
  threshold  : upper critical value (per-pixel two-tailed t-fit, alpha/2=0.025)
  sig_mask   : bool, real falls outside [lower_crit, upper_crit]
  p_values   : two-tailed p-value from per-pixel t-fit
"""

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import wandb
from matplotlib import pyplot as plt

# ── project imports (run inside container with PYTHONPATH set) ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion_brain.utils.setup import (
    load_config_from_yaml,
    parse_value,
    set_nested_attr,
)
from diffusion_brain.utils.fmri_behav_data_utils import get_roi_mask, signal_to_2d
from diffusion_brain.scripts.noise_ceiling import (
    build_repetition_groups,
    _pearsonr_columns,
    compute_noise_ceiling_1d,
)


def compute_nc_lb_avg(betas_roi, nsd_ids):
    """Slim NC-LB (averaged, leave-one-out): trial vs mean of other trials.

    Mirrors the `nc_lower_avg` block in `noise_ceiling.compute_noise_ceiling_1d`
    using the same `build_repetition_groups` / `_pearsonr_columns` helpers,
    so behaviour is identical to the existing pipeline.
    """
    groups = build_repetition_groups(nsd_ids)
    usable = {nid: idxs for nid, idxs in groups.items() if len(idxs) >= 2}
    if not usable:
        raise ValueError("No images with >=2 reps")
    n_voxels = betas_roi.shape[1]
    max_reps = max(len(idxs) for idxs in usable.values())

    r_sums = np.zeros(n_voxels, dtype=np.float64)
    count = 0
    for trial_idx in range(max_reps):
        img_ids = [nid for nid, idxs in usable.items() if len(idxs) > trial_idx]
        if len(img_ids) < 10:
            continue
        single = np.array([betas_roi[usable[nid][trial_idx]] for nid in img_ids])
        mean_others = np.array([
            np.mean([betas_roi[idx] for idx in usable[nid]
                     if idx != usable[nid][trial_idx]], axis=0)
            for nid in img_ids
        ])
        r_sums += _pearsonr_columns(single, mean_others)
        count += 1
    if count == 0:
        raise ValueError("No valid trial slots")
    return r_sums / count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Permutation test for fMRI NC-LB")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (uses data section for ROI/paths)")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--jobid", type=str, default=None)
    parser.add_argument("--n-perms", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Two-tailed alpha (default 0.05 -> alpha/2=0.025 each side)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "train", "all"])
    args_cli = parser.parse_args()

    args = load_config_from_yaml(args_cli.config)
    for ov in args_cli.override:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        set_nested_attr(args, k, parse_value(v))
        print(f"Override: {k} = {parse_value(v)}")

    subj = args.data.subj
    roi_label = f"{args.data.roi_file}_{args.data.roi}"

    run_id = args_cli.jobid or f"nc-perm-{subj}-{roi_label}"
    wandb.init(
        id=run_id,
        name=run_id,
        project="noise-ceiling-fmri",
        config={
            "subj": subj,
            "roi": args.data.roi,
            "roi_file": args.data.roi_file,
            "n_perms": args_cli.n_perms,
            "alpha": args_cli.alpha,
            "split": args_cli.split,
            "type": "permutation",
        },
    )

    print(f"\n{'='*60}")
    print(f"NC-LB permutation test for {subj}")
    print(f"ROI: {args.data.roi_file} / {args.data.roi}")
    print(f"Permutations: {args_cli.n_perms}, alpha: {args_cli.alpha}, split: {args_cli.split}")
    print(f"{'='*60}\n")

    # ── Load unaveraged single-trial betas (mirrors noise_ceiling.py) ──
    unaveraged_root = "/data/datapool3/datasets/full_nsd_betas/processed/"
    nsd_ids_path = os.path.join(unaveraged_root, subj, "nsd_ids.npy")
    betas_path = os.path.join(unaveraged_root, subj, "betas_fsaverage_unaveraged.npy")

    print(f"Loading NSD IDs from {nsd_ids_path}")
    nsd_ids = np.load(nsd_ids_path)
    print(f"  shape: {nsd_ids.shape}, unique images: {len(np.unique(nsd_ids))}")

    print(f"Loading unaveraged betas (memory-mapped)")
    full_betas = np.load(betas_path, mmap_mode="r")
    print(f"  shape: {full_betas.shape}")

    roi_indices = get_roi_mask(args)
    print(f"ROI mask: {len(roi_indices)} voxels")
    print("Loading ROI betas into memory...")
    betas_roi = np.array(full_betas[:, roi_indices])
    print(f"  ROI betas shape: {betas_roi.shape}")

    # ── Project to 2D flatmap (the representation the model operates on) ──
    print("\nProjecting ROI betas to 2D flatmap via signal_to_2d ...")
    betas_2d, locations_2d = signal_to_2d(args, data_roi=betas_roi)
    # betas_2d: (n_trials, 1, H, W); locations_2d: (y_coords, x_coords) of active pixels
    y_coords, x_coords = locations_2d
    n_active_pixels = len(y_coords)
    betas_2d_flat = betas_2d[:, 0, y_coords, x_coords].astype(np.float32)
    print(f"  2D shape: {betas_2d.shape}, active pixels: {n_active_pixels}")
    print(f"  Flattened active-pixel matrix: {betas_2d_flat.shape}")

    # ── Train/test split based on the 515 common test images ──
    test_set = set(np.load(
        os.path.join(args.data.behav_data_root, "common_515_indices.npy"),
        allow_pickle=True,
    ).tolist())

    if args_cli.split == "test":
        mask = np.array([int(n) in test_set for n in nsd_ids])
    elif args_cli.split == "train":
        mask = np.array([int(n) not in test_set for n in nsd_ids])
    else:
        mask = np.ones(len(nsd_ids), dtype=bool)

    split_betas = betas_2d_flat[mask]
    split_ids = nsd_ids[mask]
    n_trials, n_voxels = split_betas.shape
    print(f"Split '{args_cli.split}': {n_trials} trials, {n_voxels} active 2D pixels")

    # ── Real NC-LB (use full pipeline so the print summary matches noise_ceiling.py) ──
    print("\nComputing REAL NC-LB via compute_noise_ceiling_1d ...")
    real_results = compute_noise_ceiling_1d(split_betas, split_ids)
    nc_lb_real = real_results["nc_lower_avg"].astype(np.float64)
    print(f"  mean NC-LB (averaged, leave-one-out) = {nc_lb_real.mean():.4f}")

    # Sanity: slim helper must agree with the full pipeline.
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent):
        nc_lb_check = compute_nc_lb_avg(split_betas, split_ids)
    if not np.allclose(nc_lb_check, nc_lb_real, atol=1e-8):
        max_abs = np.max(np.abs(nc_lb_check - nc_lb_real))
        print(f"  WARNING: slim/full NC-LB mismatch (max abs diff = {max_abs:.2e})")

    # ── Permutation loop: shuffle image rows of [n_trials, n_voxels] ──
    print(f"\nRunning {args_cli.n_perms} permutations (shuffling betas rows)...")
    rng = np.random.default_rng(args_cli.seed)
    null_dist = np.zeros((args_cli.n_perms, n_voxels), dtype=np.float32)

    silent = io.StringIO()
    for p in range(args_cli.n_perms):
        perm_idx = rng.permutation(n_trials)
        shuffled = split_betas[perm_idx]
        with contextlib.redirect_stdout(silent):
            null_dist[p] = compute_nc_lb_avg(shuffled, split_ids)
        silent.seek(0); silent.truncate(0)

    # ── Per-voxel p=0.05 (95th percentile, one-sided) ──
    pct95 = np.percentile(null_dist, 95.0, axis=0).astype(np.float64)

    # ── Global max-stat threshold (FWER): per-perm max across voxels, take 95th ──
    perm_max = null_dist.max(axis=1)
    fwer_threshold = float(np.percentile(perm_max, 95.0))
    print(f"\n  per-voxel pct95 (mean across voxels): {pct95.mean():.4f}")
    print(f"  global FWER threshold (max-stat, alpha=0.05): {fwer_threshold:.4f}")

    # ── Two-tailed t-test per voxel: fit t to null, alpha/2 = 0.025 each side ──
    print(f"\nFitting t-distribution per voxel (two-tailed alpha={args_cli.alpha})...")
    upper_crit = np.zeros(n_voxels, dtype=np.float64)
    lower_crit = np.zeros(n_voxels, dtype=np.float64)
    p_values = np.zeros(n_voxels, dtype=np.float64)
    df_per_pixel = np.zeros(n_voxels, dtype=np.float64)
    half_alpha = args_cli.alpha / 2.0
    for v in range(n_voxels):
        col = null_dist[:, v].astype(np.float64)
        try:
            df, loc, scale = scipy.stats.t.fit(col)
        except Exception:
            df, loc, scale = 1e6, float(col.mean()), float(col.std() or 1.0)
        if not np.isfinite(scale) or scale <= 0:
            scale = max(float(col.std()), 1e-12)
        if not np.isfinite(df) or df <= 0:
            df = 1e6
        df_per_pixel[v] = df
        upper_crit[v] = scipy.stats.t.ppf(1 - half_alpha, df, loc=loc, scale=scale)
        lower_crit[v] = scipy.stats.t.ppf(half_alpha, df, loc=loc, scale=scale)
        cdf_real = scipy.stats.t.cdf(nc_lb_real[v], df, loc=loc, scale=scale)
        p_values[v] = 2.0 * min(cdf_real, 1.0 - cdf_real)

    # Empirical 97.5th / 2.5th percentile of the null per pixel — used to
    # diagnose whether the parametric t-fit reproduces the relevant tail.
    empirical_upper_crit = np.percentile(
        null_dist, 100 * (1 - half_alpha), axis=0
    ).astype(np.float64)
    empirical_lower_crit = np.percentile(
        null_dist, 100 * half_alpha, axis=0
    ).astype(np.float64)

    threshold = upper_crit
    sig_mask = (nc_lb_real > upper_crit) | (nc_lb_real < lower_crit)

    mean_nc_lb_real = float(nc_lb_real.mean())
    print(f"\n  Mean NC-LB across active pixels (real): {mean_nc_lb_real:.4f}")
    print(f"  Pixels significant (two-tailed t): {sig_mask.sum()} / {n_voxels} "
          f"({100*sig_mask.mean():.1f}%)")
    print(f"  Pixels above per-pixel pct95:      {(nc_lb_real > pct95).sum()} / {n_voxels} "
          f"({100*(nc_lb_real > pct95).mean():.1f}%)")
    print(f"  Pixels above FWER threshold:       {(nc_lb_real > fwer_threshold).sum()} / {n_voxels} "
          f"({100*(nc_lb_real > fwer_threshold).mean():.1f}%)")

    # ── wandb logging: null distribution histograms ──
    print("\nLogging null distribution histograms to wandb...")
    payload = {}

    # (1) Mean null distribution across pixels: per-perm mean across pixels
    null_mean_across_pixels = null_dist.mean(axis=1)  # (n_perms,)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_mean_across_pixels, bins=80, alpha=0.7, color="steelblue",
            density=True, label="null mean NC-LB (per perm, mean over pixels)")
    ax.axvline(mean_nc_lb_real, color="firebrick", linewidth=2,
               label=f"real mean = {mean_nc_lb_real:.4f}")
    ax.set_xlabel("mean NC-LB across active pixels")
    ax.set_ylabel("density")
    ax.set_title(f"Null distribution of mean NC-LB ({subj}, {roi_label}, split={args_cli.split})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    payload["null_hist_mean_across_pixels"] = wandb.Image(fig)
    plt.close(fig)

    # (2) Aggregate null + real overlay (kept for reference)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_dist.ravel(), bins=200, alpha=0.6, color="steelblue",
            density=True, label="null NC-LB (all pixels x perms)")
    ax.hist(nc_lb_real, bins=100, alpha=0.5, color="firebrick",
            density=True, label="real NC-LB (per pixel)")
    ax.axvline(fwer_threshold, color="black", linestyle="--",
               label=f"FWER thr = {fwer_threshold:.3f}")
    ax.set_xlabel("NC-LB (Pearson r)")
    ax.set_ylabel("density")
    ax.set_title(f"Null vs real NC-LB aggregate ({subj}, {roi_label}, split={args_cli.split})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    payload["null_hist_aggregate"] = wandb.Image(fig)
    plt.close(fig)

    # (3) Null histograms for the 3 most significant pixels (smallest p-value,
    #     ties broken by largest real NC-LB)
    sort_key = np.lexsort((-nc_lb_real, p_values))  # ascending p, then descending real
    top3 = sort_key[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, v in zip(axes, top3):
        ax.hist(null_dist[:, v], bins=60, color="steelblue",
                alpha=0.7, density=True, label="null")
        ax.axvline(nc_lb_real[v], color="firebrick", linewidth=2,
                   label=f"real={nc_lb_real[v]:.3f}")
        ax.axvline(upper_crit[v], color="black", linestyle="--",
                   label=f"upper={upper_crit[v]:.3f}")
        ax.axvline(lower_crit[v], color="black", linestyle=":",
                   label=f"lower={lower_crit[v]:.3f}")
        ax.set_title(f"pixel {v}  p={p_values[v]:.3g}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    payload["null_hist_top3_significant"] = wandb.Image(fig)
    plt.close(fig)

    # Interactive wandb histogram on a subsample of the null distribution
    sample = null_dist.ravel()
    if sample.size > 200_000:
        sub_rng = np.random.default_rng(args_cli.seed + 1)
        sample = sample[sub_rng.choice(sample.size, 200_000, replace=False)]
    table = wandb.Table(data=[[float(v)] for v in sample], columns=["nc_lb_null"])
    payload["null_hist_interactive"] = wandb.plot.histogram(
        table, value="nc_lb_null", title="null NC-LB (sampled across voxels x perms)"
    )

    # ── t-fit diagnostics ──
    # (a) Scatter: parametric upper crit vs empirical 97.5th percentile across
    #     all pixels. Closer to y=x ⇒ t-fit reproduces the relevant tail.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(empirical_upper_crit, upper_crit, s=4, alpha=0.4,
               color="steelblue", label="pixels")
    lo = float(min(empirical_upper_crit.min(), upper_crit.min()))
    hi = float(max(empirical_upper_crit.max(), upper_crit.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", label="y = x")
    ax.set_xlabel("empirical 97.5th percentile of null")
    ax.set_ylabel("parametric t-fit upper crit (97.5%)")
    ax.set_title(f"t-fit tail diagnostic ({subj}, {roi_label}, split={args_cli.split})")
    # Quantitative signed bias (parametric - empirical): tells you whether the
    # t-fit is systematically liberal (>0) or conservative (<0) at the cutoff.
    diff_upper = upper_crit - empirical_upper_crit
    median_bias = float(np.median(diff_upper))
    p95_abs_bias = float(np.percentile(np.abs(diff_upper), 95))
    ax.text(0.05, 0.95,
            f"median(t - emp) = {median_bias:+.4f}\n"
            f"95% |t - emp|  ≤ {p95_abs_bias:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    payload["t_fit_upper_crit_vs_empirical"] = wandb.Image(fig)
    plt.close(fig)

    # (b) Histogram of fitted df values per pixel. Large df ⇒ collapses to
    #     Gaussian; small df ⇒ heavy-tailed t doing real work.
    fig, ax = plt.subplots(figsize=(8, 5))
    df_clipped = np.clip(df_per_pixel, 0, 200)  # cap visualisation; t fits sometimes return huge df
    ax.hist(df_clipped, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(np.median(df_per_pixel), color="firebrick", linewidth=2,
               label=f"median df = {np.median(df_per_pixel):.1f}")
    ax.set_xlabel("fitted df (clipped at 200 for display)")
    ax.set_ylabel("# pixels")
    ax.set_title(f"Per-pixel t-fit df ({subj}, {roi_label}, split={args_cli.split})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    payload["t_fit_df_histogram"] = wandb.Image(fig)
    plt.close(fig)

    payload["t_fit_upper_crit_median_bias"] = median_bias
    payload["t_fit_upper_crit_p95_abs_bias"] = p95_abs_bias
    payload["t_fit_df_median"] = float(np.median(df_per_pixel))
    payload["t_fit_df_p10"] = float(np.percentile(df_per_pixel, 10))

    # Scalar summaries
    payload["mean_nc_lb_real"] = mean_nc_lb_real
    payload["mean_nc_lb_null"] = float(null_mean_across_pixels.mean())
    payload["mean_nc_lb_null_pct95"] = float(np.percentile(null_mean_across_pixels, 95.0))
    payload["fwer_threshold"] = fwer_threshold
    payload["frac_sig_two_tailed_t"] = float(sig_mask.mean())
    payload["frac_sig_per_pixel_pct95"] = float((nc_lb_real > pct95).mean())
    payload["frac_sig_fwer"] = float((nc_lb_real > fwer_threshold).mean())
    payload["n_sig_two_tailed_t"] = int(sig_mask.sum())
    payload["n_sig_per_pixel_pct95"] = int((nc_lb_real > pct95).sum())
    payload["n_sig_fwer"] = int((nc_lb_real > fwer_threshold).sum())

    # Brain maps in 2D pixel space — scatter pixel values back into the grid
    H, W = betas_2d.shape[2], betas_2d.shape[3]

    def _scatter_to_2d(values_per_pixel):
        grid = np.zeros((H, W), dtype=np.float32)
        grid[y_coords, x_coords] = values_per_pixel
        return grid

    from matplotlib.colors import TwoSlopeNorm
    real_grid = _scatter_to_2d(np.clip(nc_lb_real, 0, 1))
    fig, ax = plt.subplots()
    abs_max = max(abs(real_grid.min()), abs(real_grid.max()), 1e-8)
    im = ax.imshow(real_grid, cmap="RdBu_r", origin="lower",
                   norm=TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max))
    ax.set_title(f"Real NC-LB 2D ({subj}, split={args_cli.split})")
    plt.colorbar(im, ax=ax)
    payload["real_nc_lb_2d"] = wandb.Image(fig)
    plt.close(fig)

    sig_grid = _scatter_to_2d(sig_mask.astype(np.float32))
    fig, ax = plt.subplots()
    im = ax.imshow(sig_grid, cmap="RdBu_r", origin="lower", vmin=-1, vmax=1)
    ax.set_title(f"Sig mask 2D (two-tailed t, alpha={args_cli.alpha})")
    plt.colorbar(im, ax=ax)
    payload["sig_mask_two_tailed_t_2d"] = wandb.Image(fig)
    plt.close(fig)

    wandb.log(payload)

    # ── Save results ──
    if args_cli.output:
        out_path = args_cli.output
    else:
        out_dir = os.path.join("results", "noise_ceiling")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"noise_ceiling_perm_{subj}_{args.data.roi_file}_"
            f"{args.data.roi}_{args_cli.split}.npz",
        )

    np.savez(
        out_path,
        nc_lb_real=nc_lb_real,
        threshold=threshold,
        sig_mask=sig_mask,
        p_values=p_values,
        upper_crit=upper_crit,
        lower_crit=lower_crit,
        empirical_upper_crit=empirical_upper_crit,
        empirical_lower_crit=empirical_lower_crit,
        df_per_pixel=df_per_pixel,
        pct95_per_pixel=pct95,
        fwer_threshold=fwer_threshold,
        # 2D-pixel-space metadata
        pixel_y_coords=y_coords,
        pixel_x_coords=x_coords,
        grid_h=H,
        grid_w=W,
        roi_indices=roi_indices,
        n_perms=args_cli.n_perms,
        alpha=args_cli.alpha,
        seed=args_cli.seed,
        subj=subj,
        roi=args.data.roi,
        roi_file=args.data.roi_file,
        split=args_cli.split,
    )
    print(f"\nResults saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
