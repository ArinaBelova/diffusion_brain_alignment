"""Scatter plot: encoding model performance vs interparticipant agreement.

Each dot = one voxel (2D flatmap pixel).
  x-axis: interparticipant agreement (leave-one-out noise ceiling)
  y-axis: encoding model per-voxel r (group mean across subjects)

Overlays diffusion and ridge results on the same axes.

The noise ceiling is computed on all ROI vertices in fsaverage space (e.g.
19065), but the encoding models operate on 2D flatmap pixels (e.g. 16186)
because the projection averages colliding vertices. This script projects
the noise ceiling through the same 2D grid via signal_to_2d, then extracts
at the ROI pixel locations so all arrays are aligned.

In addition to the scatter plot, this script performs a paired subject-wise
significance analysis on the per-voxel r-score difference (diffusion − ridge):
  * 2-tailed paired t-test on (r_diffusion − r_ridge) across subjects, per voxel
  * Benjamini–Hochberg FDR correction at q < 0.05
  * Brain-mapped outputs (mean difference, FDR-thresholded mean difference,
    significance mask, t-stats, -log10 p_corrected) saved to disk and logged
    to wandb. Positive values = diffusion beats ridge.
The paired analysis requires both --diffusion and --ridge.

Usage (inside container):
    python figures/plot_encoding_vs_noise_ceiling.py \
        --config src/diffusion_brain/configs/brain/2d_config_generate.yaml \
        --noise-ceiling results/noise_ceiling/noise_ceiling_intersubject_streams_5.npz \
        --diffusion outputs/generated_samples/ann-brain/<jobid>/group_generation_results_best.npz \
        --ridge outputs/ann-brain/<jobid>/group_ridge_results.npz \
        --output figures/encoding_vs_noise_ceiling.pdf
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import wandb
from scipy.stats import ttest_1samp, wilcoxon

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_brain.utils.setup import load_config_from_yaml, parse_value, set_nested_attr
from diffusion_brain.utils.fmri_behav_data_utils import signal_to_2d
from diffusion_brain.utils.nc_correction import benjamini_hochberg, build_brain_map
from diffusion_brain.utils.visualise import fmri_to_wandb_image

FDR_ALPHA = 0.05


def _project_nc_to_2d(nc_per_vertex, cfg):
    """Project per-vertex noise ceiling through signal_to_2d.

    Returns (nc_2d_image, locations_from_projection) — `nc_2d_image` has
    shape (H, W) where (H, W) is the cropped flatmap size used by every
    encoding model in this project.
    """
    nc_2d, locations_from_proj = signal_to_2d(cfg, one_signal_to_transform=nc_per_vertex)
    nc_2d = nc_2d.squeeze()  # (H, W)
    return nc_2d, locations_from_proj


def load_encoding_r(path, label):
    """Load group_mean_r (n_voxels,) from a results npz."""
    data = np.load(path, allow_pickle=True)
    print(f"Loaded {label} encoding results from {path}. Keys: {list(data.keys())}", flush=True)
    if "group_mean_r" in data:
        return data["group_mean_r"]
    elif "all_r" in data:
        return np.nanmean(data["all_r"], axis=0)
    else:
        raise KeyError(f"{label} npz has no 'group_mean_r' or 'all_r' key. Keys: {list(data.keys())}")


def load_subject_wise_r(path, label):
    """Load per-subject per-voxel r matrix (N_subj, n_voxels) from a results npz."""
    data = np.load(path, allow_pickle=True)
    if "all_r" not in data:
        raise KeyError(
            f"{label} npz has no 'all_r' key — subject-wise significance analysis "
            f"requires the per-subject per-voxel r matrix. Keys: {list(data.keys())}"
        )
    all_r = np.asarray(data["all_r"])
    if all_r.ndim != 2:
        raise ValueError(f"{label}: 'all_r' must be 2D (n_subj, n_voxels), got shape {all_r.shape}")
    subj_names = data["subject_names"].tolist() if "subject_names" in data else None
    return all_r, subj_names


def subject_wise_significance(all_r, alpha=FDR_ALPHA):
    """2-tailed t-test vs 0 over subjects + BH-FDR correction.

    Parameters
    ----------
    all_r : ndarray (n_subj, n_voxels)
        Per-subject per-voxel Pearson r-scores.
    alpha : float
        BH-FDR target false-discovery rate.

    Returns
    -------
    dict with keys:
        group_mean_r : (n_voxels,)
        t_stats, p_values, p_corrected : (n_voxels,)
        reject : (n_voxels,) boolean — voxels surviving BH-FDR at q<alpha
        n_subjects, n_voxels, n_significant, frac_significant : scalars
        mean_r_all, mean_r_significant : scalars
    """
    all_r = np.asarray(all_r, dtype=np.float64)
    n_subjects, n_voxels = all_r.shape

    group_mean_r = np.nanmean(all_r, axis=0)

    t_stats, p_values = ttest_1samp(all_r, popmean=0.0, axis=0, nan_policy="omit")
    t_stats = np.asarray(t_stats, dtype=np.float64)
    p_values = np.asarray(p_values, dtype=np.float64)
    nan_mask = np.isnan(p_values)
    if nan_mask.any():
        p_values[nan_mask] = 1.0

    reject, p_corrected = benjamini_hochberg(p_values, alpha=alpha)
    n_sig = int(reject.sum())
    mean_r_all = float(np.nanmean(group_mean_r))
    mean_r_sig = float(np.nanmean(group_mean_r[reject])) if n_sig > 0 else 0.0

    return {
        "group_mean_r": group_mean_r,
        "t_stats": t_stats,
        "p_values": p_values,
        "p_corrected": p_corrected,
        "reject": reject,
        "n_subjects": int(n_subjects),
        "n_voxels": int(n_voxels),
        "n_significant": n_sig,
        "n_uncorrected_significant": int((p_values < alpha).sum()),
        "frac_significant": float(n_sig / n_voxels) if n_voxels > 0 else 0.0,
        "mean_r_all": mean_r_all,
        "mean_r_significant": mean_r_sig,
        "alpha": float(alpha),
    }


def wilcoxon_significance(delta_r, alpha=FDR_ALPHA):
    """2-tailed Wilcoxon signed-rank vs 0 over subjects + BH-FDR correction.

    Non-parametric counterpart to `subject_wise_significance`: ranks |delta|
    per voxel and tests whether positive and negative differences balance.
    No normality assumption — preferable for n=8 paired differences.

    Parameters
    ----------
    delta_r : ndarray (n_subj, n_voxels)
        Per-subject per-voxel difference (e.g. r_diffusion − r_ridge).
    alpha : float
        BH-FDR target false-discovery rate.

    Returns
    -------
    dict mirroring `subject_wise_significance`, but with:
        median_delta_r : (n_voxels,)    — central-tendency estimator the
                                          signed-rank test implicitly tests.
        W_stats        : (n_voxels,)    — sum of positive ranks.
        median_delta_all / median_delta_significant : scalars.
    """
    delta_r = np.asarray(delta_r, dtype=np.float64)
    n_subjects, n_voxels = delta_r.shape

    median_delta = np.nanmedian(delta_r, axis=0)

    # Column-by-column loop. Vectorised wilcoxon (scipy >=1.9) is faster but
    # raises ValueError on all-zero or near-degenerate columns; iterating
    # keeps NaN handling and degenerate columns under our control.
    W_stats = np.full(n_voxels, np.nan, dtype=np.float64)
    p_values = np.ones(n_voxels, dtype=np.float64)
    for v in range(n_voxels):
        col = delta_r[:, v]
        col = col[np.isfinite(col)]
        if col.size < 1 or np.all(col == 0):
            continue
        try:
            res = wilcoxon(col, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            continue
        W_stats[v] = float(res.statistic)
        p_values[v] = float(res.pvalue)

    nan_mask = np.isnan(p_values)
    if nan_mask.any():
        p_values[nan_mask] = 1.0

    reject, p_corrected = benjamini_hochberg(p_values, alpha=alpha)
    print("p_values stats: min p_value = {:.2e}, max p_value = {:.2e}".format(
        np.min(p_values), np.max(p_values)
    ), flush=True)
    print("p_corrected stats: min p_corrected = {:.2e}, max p_corrected = {:.2e}".format(
        np.min(p_corrected), np.max(p_corrected)
    ), flush=True)
    n_sig = int(reject.sum())
    n_uncorrected = int((p_values < alpha).sum())

    med_all = float(np.nanmean(median_delta))
    med_sig = float(np.nanmean(median_delta[reject])) if n_sig > 0 else 0.0

    # With n paired observations the smallest possible 2-sided exact p-value
    # is 2/2^n (all same sign). Reported so the caller can see the floor.
    p_floor = 2.0 / (2 ** n_subjects)

    return {
        "median_delta_r": median_delta,
        "W_stats": W_stats,
        "p_values": p_values,
        "p_corrected": p_corrected,
        "reject": reject,
        "n_subjects": int(n_subjects),
        "n_voxels": int(n_voxels),
        "n_significant": n_sig,
        "n_uncorrected_significant": n_uncorrected,
        "frac_significant": float(n_sig / n_voxels) if n_voxels > 0 else 0.0,
        "median_delta_all": med_all,
        "median_delta_significant": med_sig,
        "p_value_floor": p_floor,
        "alpha": float(alpha),
    }


def build_wilcoxon_brain_maps(stats, image_shape, y_coords, x_coords):
    """Brain maps for Wilcoxon results: median Δ, sig-only median Δ, W, sig mask, -log10 p_corr."""
    med = stats["median_delta_r"]
    reject = stats["reject"]

    med_all_img = build_brain_map(med, image_shape, y_coords, x_coords)

    sig_values = np.where(reject, np.nan_to_num(med, nan=0.0), 0.0)
    med_sig_img = build_brain_map(sig_values, image_shape, y_coords, x_coords)

    sig_mask_img = build_brain_map(reject.astype(np.float64), image_shape, y_coords, x_coords)

    W_img = build_brain_map(stats["W_stats"], image_shape, y_coords, x_coords)

    p_c = np.clip(stats["p_corrected"], 1e-300, 1.0)
    neg_log10_p_img = build_brain_map(-np.log10(p_c), image_shape, y_coords, x_coords)

    p_raw = np.clip(stats["p_values"], 1e-300, 1.0)
    neg_log10_p_raw_img = build_brain_map(-np.log10(p_raw), image_shape, y_coords, x_coords)

    return {
        "median_delta_image": med_all_img,
        "median_delta_significant_image": med_sig_img,
        "significance_mask_image": sig_mask_img,
        "W_stats_image": W_img,
        "neg_log10_p_corrected_image": neg_log10_p_img,
        "neg_log10_p_uncorrected_image": neg_log10_p_raw_img,
    }


def build_significance_brain_maps(stats, image_shape, y_coords, x_coords):
    """Place per-voxel arrays from `subject_wise_significance` onto the 2D flatmap.

    Returns a dict of (H, W) arrays for: group_mean_r, group_mean_r_significant
    (BH-FDR mask), group_mean_r_uncorrected (p < alpha mask), significance_mask,
    t_stats, neg_log10_p_corrected, neg_log10_p_uncorrected.
    """
    gmr = stats["group_mean_r"]
    reject = stats["reject"]
    alpha = stats["alpha"]
    p_values = stats["p_values"]

    gmr_all_img = build_brain_map(gmr, image_shape, y_coords, x_coords)

    sig_values = np.where(reject, np.nan_to_num(gmr, nan=0.0), 0.0)
    gmr_sig_img = build_brain_map(sig_values, image_shape, y_coords, x_coords)

    # Uncorrected mask: per-voxel p < alpha, no multiple-comparisons correction.
    uncorr_mask = p_values < alpha
    uncorr_values = np.where(uncorr_mask, np.nan_to_num(gmr, nan=0.0), 0.0)
    gmr_uncorr_img = build_brain_map(uncorr_values, image_shape, y_coords, x_coords)

    sig_mask_img = build_brain_map(reject.astype(np.float64), image_shape, y_coords, x_coords)

    t_img = build_brain_map(stats["t_stats"], image_shape, y_coords, x_coords)

    # Cap -log10(p) at a large but finite value so colormap doesn't explode.
    p_c = np.clip(stats["p_corrected"], 1e-300, 1.0)
    neg_log10_p = -np.log10(p_c)
    neg_log10_p_img = build_brain_map(neg_log10_p, image_shape, y_coords, x_coords)

    p_raw = np.clip(p_values, 1e-300, 1.0)
    neg_log10_p_raw_img = build_brain_map(-np.log10(p_raw), image_shape, y_coords, x_coords)


    return {
        "group_mean_r_image": gmr_all_img,
        "group_mean_r_significant_image": gmr_sig_img,
        "group_mean_r_uncorrected_image": gmr_uncorr_img,
        "significance_mask_image": sig_mask_img,
        "t_stats_image": t_img,
        "neg_log10_p_corrected_image": neg_log10_p_img,
        "neg_log10_p_uncorrected_image": neg_log10_p_raw_img,
    }


def _align_subjects(diff_entry, ridge_entry):
    """Align two (n_subj, n_voxels) matrices by subject name.

    Returns (diff_aligned, ridge_aligned, subjects_in_order) where rows in
    the two matrices correspond to the same subject. If either entry is
    missing subject names, falls back to assuming the rows are already in
    the same order (and asserts equal n_subj).
    """
    diff_all_r = diff_entry["all_r"]
    ridge_all_r = ridge_entry["all_r"]
    diff_names = diff_entry["subj_names"]
    ridge_names = ridge_entry["subj_names"]

    if diff_names is None or ridge_names is None:
        if diff_all_r.shape[0] != ridge_all_r.shape[0]:
            raise ValueError(
                f"Cannot align subjects without names: diffusion has "
                f"{diff_all_r.shape[0]} rows but ridge has {ridge_all_r.shape[0]}."
            )
        order = [f"subj_{i}" for i in range(diff_all_r.shape[0])]
        print(
            "WARNING: subject_names absent from one of the npz files — "
            "assuming row order matches.",
            flush=True,
        )
        return diff_all_r, ridge_all_r, order

    diff_names = [str(s) for s in diff_names]
    ridge_names = [str(s) for s in ridge_names]
    common = [s for s in diff_names if s in set(ridge_names)]
    if not common:
        raise ValueError(
            f"No overlapping subjects between diffusion ({diff_names}) and ridge "
            f"({ridge_names})."
        )
    diff_idx = [diff_names.index(s) for s in common]
    ridge_idx = [ridge_names.index(s) for s in common]
    return diff_all_r[diff_idx], ridge_all_r[ridge_idx], common


def _run_diffusion_minus_ridge_analysis(
    per_model_all_r, image_shape, y_coords, x_coords, fdr_alpha, output_dir,
):
    """Paired per-voxel t-test on (r_diffusion − r_ridge) + BH-FDR + brain maps.

    Logs the maps to wandb under `significance/diffusion_minus_ridge/...`
    and writes `significance_diffusion_minus_ridge.npz` next to the figure.
    Returns a dict with `stats`, `maps`, and `npz_path`, or None if the
    analysis was skipped (e.g. only one model provided).
    """
    diff_entry = per_model_all_r.get("Diffusion model")
    ridge_entry = per_model_all_r.get("Ridge regression")
    if diff_entry is None or ridge_entry is None:
        print(
            "Skipping diffusion − ridge paired analysis: both --diffusion and "
            "--ridge are required (and both must expose `all_r`).",
            flush=True,
        )
        return None

    diff_r, ridge_r, common_subjects = _align_subjects(diff_entry, ridge_entry)
    print(
        f"\nPaired diffusion − ridge analysis on {len(common_subjects)} subjects: "
        f"{common_subjects}",
        flush=True,
    )

    # Per-voxel signed difference across subjects, shape (n_subj, n_voxels).
    # Two parallel tests are run on it:
    #   t  : parametric paired t-test vs 0 (`subject_wise_significance`)
    #   W  : non-parametric Wilcoxon signed-rank vs 0 (`wilcoxon_significance`)
    delta_r = diff_r - ridge_r

    stats_t = subject_wise_significance(delta_r, alpha=fdr_alpha)
    maps_t = build_significance_brain_maps(stats_t, image_shape, y_coords, x_coords)

    stats_w = wilcoxon_significance(delta_r, alpha=fdr_alpha)
    maps_w = build_wilcoxon_brain_maps(stats_w, image_shape, y_coords, x_coords)

    # ── Console summaries ──
    n_pos_t = int(np.sum(stats_t["reject"] & (stats_t["group_mean_r"] > 0)))
    n_neg_t = int(np.sum(stats_t["reject"] & (stats_t["group_mean_r"] < 0)))
    n_pos_w = int(np.sum(stats_w["reject"] & (stats_w["median_delta_r"] > 0)))
    n_neg_w = int(np.sum(stats_w["reject"] & (stats_w["median_delta_r"] < 0)))

    print(f"{'-' * 60}", flush=True)
    print(f"Diffusion − Ridge: paired t-test + BH-FDR q<{stats_t['alpha']}", flush=True)
    print(f"  n_subjects                            = {stats_t['n_subjects']}", flush=True)
    print(f"  mean(Δr) all voxels                   = {stats_t['mean_r_all']:+.4f}", flush=True)
    print(f"  mean(Δr) BH-FDR-significant voxels    = {stats_t['mean_r_significant']:+.4f}", flush=True)
    print(
        f"  significant voxels (BH-FDR)           = {stats_t['n_significant']}/"
        f"{stats_t['n_voxels']} ({100.0 * stats_t['frac_significant']:.1f}%)",
        flush=True,
    )
    print(
        f"    diffusion > ridge / ridge > diffusion = {n_pos_t} / {n_neg_t}",
        flush=True,
    )
    print(
        f"  uncorrected p<{stats_t['alpha']} voxels        = "
        f"{stats_t['n_uncorrected_significant']}/{stats_t['n_voxels']}",
        flush=True,
    )
    print(f"{'-' * 60}", flush=True)
    print(f"Diffusion − Ridge: Wilcoxon signed-rank + BH-FDR q<{stats_w['alpha']}", flush=True)
    print(f"  n_subjects                            = {stats_w['n_subjects']}", flush=True)
    print(
        f"  smallest possible 2-sided p (n={stats_w['n_subjects']}) "
        f"= 2/2^n = {stats_w['p_value_floor']:.4f}",
        flush=True,
    )
    print(
        f"  BH threshold for smallest p           ≈ {fdr_alpha / max(stats_w['n_voxels'], 1):.2e}",
        flush=True,
    )
    print(f"  median(Δr) all voxels                 = {stats_w['median_delta_all']:+.4f}", flush=True)
    print(f"  median(Δr) BH-FDR-significant         = {stats_w['median_delta_significant']:+.4f}", flush=True)
    print(
        f"  significant voxels (BH-FDR)           = {stats_w['n_significant']}/"
        f"{stats_w['n_voxels']} ({100.0 * stats_w['frac_significant']:.1f}%)",
        flush=True,
    )
    print(
        f"    diffusion > ridge / ridge > diffusion = {n_pos_w} / {n_neg_w}",
        flush=True,
    )
    print(
        f"  uncorrected p<{stats_w['alpha']} voxels        = "
        f"{stats_w['n_uncorrected_significant']}/{stats_w['n_voxels']}",
        flush=True,
    )
    print(f"{'-' * 60}\n", flush=True)

    # ── wandb: t-test maps ──
    prefix_t = "significance/diffusion_minus_ridge/t"
    wandb.log({
        f"{prefix_t}/mean_delta_r_image": fmri_to_wandb_image(
            maps_t["group_mean_r_image"],
            title=f"t-test: mean Δr (all voxels, n_subj={stats_t['n_subjects']})",
            robust=True,
        ),
        f"{prefix_t}/mean_delta_r_significant_image": fmri_to_wandb_image(
            maps_t["group_mean_r_significant_image"],
            title=(
                f"t-test: mean Δr, BH-FDR q<{stats_t['alpha']} "
                f"({stats_t['n_significant']}/{stats_t['n_voxels']} sig)"
            ),
            robust=True,
        ),
        f"{prefix_t}/mean_delta_r_uncorrected_image": fmri_to_wandb_image(
            maps_t["group_mean_r_uncorrected_image"],
            title=(
                f"t-test: mean Δr, uncorrected p<{stats_t['alpha']} "
                f"({stats_t['n_uncorrected_significant']}/{stats_t['n_voxels']} sig)"
            ),
            robust=True,
        ),
        f"{prefix_t}/significance_mask_image": fmri_to_wandb_image(
            maps_t["significance_mask_image"],
            title=f"t-test BH-FDR significance mask (q<{stats_t['alpha']})",
        ),
        f"{prefix_t}/t_stats_image": fmri_to_wandb_image(
            maps_t["t_stats_image"],
            title=f"Paired t-statistic (n_subj={stats_t['n_subjects']})",
            robust=True,
        ),
        f"{prefix_t}/neg_log10_p_corrected_image": fmri_to_wandb_image(
            maps_t["neg_log10_p_corrected_image"],
            title="t-test: -log10(BH-corrected p)",
            robust=True,
        ),
        f"{prefix_t}/n_subjects": stats_t["n_subjects"],
        f"{prefix_t}/n_voxels": stats_t["n_voxels"],
        f"{prefix_t}/n_significant": stats_t["n_significant"],
        f"{prefix_t}/n_uncorrected_significant": stats_t["n_uncorrected_significant"],
        f"{prefix_t}/frac_significant": stats_t["frac_significant"],
        f"{prefix_t}/n_significant_diffusion_better": n_pos_t,
        f"{prefix_t}/n_significant_ridge_better": n_neg_t,
        f"{prefix_t}/mean_delta_r_all_voxels": stats_t["mean_r_all"],
        f"{prefix_t}/mean_delta_r_significant": stats_t["mean_r_significant"],
    })

    # ── wandb: Wilcoxon maps ──
    prefix_w = "significance/diffusion_minus_ridge/wilcoxon"
    wandb.log({
        f"{prefix_w}/median_delta_image": fmri_to_wandb_image(
            maps_w["median_delta_image"],
            title=f"Wilcoxon: median Δr (all voxels, n_subj={stats_w['n_subjects']})",
            robust=True,
        ),
        f"{prefix_w}/median_delta_significant_image": fmri_to_wandb_image(
            maps_w["median_delta_significant_image"],
            title=(
                f"Wilcoxon: median Δr, BH-FDR q<{stats_w['alpha']} "
                f"({stats_w['n_significant']}/{stats_w['n_voxels']} sig)"
            ),
            robust=True,
        ),
        f"{prefix_w}/significance_mask_image": fmri_to_wandb_image(
            maps_w["significance_mask_image"],
            title=f"Wilcoxon BH-FDR significance mask (q<{stats_w['alpha']})",
        ),
        f"{prefix_w}/W_stats_image": fmri_to_wandb_image(
            maps_w["W_stats_image"],
            title=f"Wilcoxon W (sum of positive ranks, n_subj={stats_w['n_subjects']})",
            robust=True,
        ),
        f"{prefix_w}/neg_log10_p_corrected_image": fmri_to_wandb_image(
            maps_w["neg_log10_p_corrected_image"],
            title="Wilcoxon: -log10(BH-corrected p)",
            robust=True,
        ),
        f"{prefix_w}/neg_log10_p_uncorrected_image": fmri_to_wandb_image(
            maps_w["neg_log10_p_uncorrected_image"],
            title="Wilcoxon: -log10(uncorrected p)",
            robust=True,
        ),
        f"{prefix_w}/n_subjects": stats_w["n_subjects"],
        f"{prefix_w}/n_voxels": stats_w["n_voxels"],
        f"{prefix_w}/n_significant": stats_w["n_significant"],
        f"{prefix_w}/n_uncorrected_significant": stats_w["n_uncorrected_significant"],
        f"{prefix_w}/frac_significant": stats_w["frac_significant"],
        f"{prefix_w}/n_significant_diffusion_better": n_pos_w,
        f"{prefix_w}/n_significant_ridge_better": n_neg_w,
        f"{prefix_w}/median_delta_all": stats_w["median_delta_all"],
        f"{prefix_w}/median_delta_significant": stats_w["median_delta_significant"],
        f"{prefix_w}/p_value_floor": stats_w["p_value_floor"],
    })

    # ── npz: t-test fields keep their existing names; Wilcoxon fields go
    # under `wilcoxon_*` so the file is additive-only and back-compatible.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "significance_diffusion_minus_ridge.npz"
    save_dict = {
        "delta_r_per_subject": delta_r,
        "diffusion_r_per_subject": diff_r,
        "ridge_r_per_subject": ridge_r,
        "subject_names": np.array(common_subjects),
        "mean_delta_r": stats_t["group_mean_r"],
        "t_stats": stats_t["t_stats"],
        "p_values": stats_t["p_values"],
        "p_corrected": stats_t["p_corrected"],
        "reject": stats_t["reject"],
        "alpha": stats_t["alpha"],
        "y_coords": y_coords,
        "x_coords": x_coords,
        "image_shape": np.array(image_shape),
        **maps_t,
        "wilcoxon_median_delta_r": stats_w["median_delta_r"],
        "wilcoxon_W_stats": stats_w["W_stats"],
        "wilcoxon_p_values": stats_w["p_values"],
        "wilcoxon_p_corrected": stats_w["p_corrected"],
        "wilcoxon_reject": stats_w["reject"],
        "wilcoxon_p_value_floor": stats_w["p_value_floor"],
        **{f"wilcoxon_{k}": v for k, v in maps_w.items()},
    }
    np.savez(npz_path, **save_dict)
    print(f"Saved diffusion − ridge significance maps → {npz_path}", flush=True)

    return {
        "t": {"stats": stats_t, "maps": maps_t},
        "wilcoxon": {"stats": stats_w, "maps": maps_w},
        "npz_path": str(npz_path),
        "subjects": common_subjects,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Encoding model performance vs interparticipant agreement scatter plot"
    )
    parser.add_argument("--config", required=True, help="YAML config (for ROI geometry / signal_to_2d)")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--noise-ceiling", required=True, help="Path to noise_ceiling_intersubject_*.npz")
    parser.add_argument("--diffusion", default=None, help="Path to group_generation_results_*.npz")
    parser.add_argument("--ridge", default=None, help="Path to group_ridge_results.npz")
    parser.add_argument("--output", default="figures/encoding_vs_noise_ceiling.pdf", help="Output figure path")
    parser.add_argument("--alpha", type=float, default=0.15, help="Scatter point transparency")
    parser.add_argument("--point-size", type=float, default=4.0, help="Scatter point size")
    parser.add_argument(
        "--fdr-alpha", type=float, default=FDR_ALPHA,
        help="Benjamini–Hochberg target FDR for the diffusion − ridge difference analysis",
    )
    parser.add_argument(
        "--wandb-project", default="encoding-vs-noise-ceiling",
        help="wandb project for significance brain maps",
    )
    parser.add_argument(
        "--wandb-run-name", default=None,
        help="wandb run name (defaults to a name derived from --output)",
    )
    parser.add_argument(
        "--wandb-mode", default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode — set to 'disabled' to skip wandb logging entirely",
    )
    cli = parser.parse_args()

    if cli.diffusion is None and cli.ridge is None:
        parser.error("Provide at least one of --diffusion or --ridge")

    # Load config for ROI geometry
    cfg = load_config_from_yaml(cli.config)
    for override in cli.override:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        set_nested_attr(cfg, key, parse_value(value))

    # Load noise ceiling (per-vertex, e.g. 19065)
    nc = np.load(cli.noise_ceiling, allow_pickle=True)
    nc_r_vertex = nc["avg_r_lower"]
    print(f"Noise ceiling: {nc_r_vertex.shape[0]} vertices (roi_indices space)", flush=True)

    # Project noise ceiling to a 2D flatmap and extract at the ROI pixel locations.
    # `(y_coords, x_coords)` are the same 2D pixel indices used by every encoding
    # model in this project (ridge + diffusion), and image_shape = nc_2d.shape
    # is therefore the same flatmap geometry used by generate.py for brain maps.
    nc_2d, _ = _project_nc_to_2d(nc_r_vertex, cfg)
    image_shape = nc_2d.shape

    roi_locations_path = os.path.join(
        cfg.data.roi_defs_dir,
        f"roi_preselected_extended_2d_images_res_{cfg.data.grid_resolution_2d}",
        f"{cfg.data.roi_file}",
        f"{cfg.data.subj}_{cfg.data.roi}.npz",
    )
    loc = np.load(roi_locations_path, allow_pickle=True)["locations"]
    y_coords, x_coords = loc[0], loc[1]
    nc_r = nc_2d[y_coords, x_coords]
    n_pixels = len(nc_r)
    print(f"Noise ceiling projected to 2D: {n_pixels} pixels", flush=True)

    # ── wandb init for significance brain maps ──
    run_name = cli.wandb_run_name or Path(cli.output).stem
    wandb.init(
        project=cli.wandb_project,
        name=run_name,
        mode=cli.wandb_mode,
        config={
            "roi": getattr(cfg.data, "roi", None),
            "roi_file": getattr(cfg.data, "roi_file", None),
            "subj": getattr(cfg.data, "subj", None),
            "grid_resolution_2d": getattr(cfg.data, "grid_resolution_2d", None),
            "fdr_alpha": cli.fdr_alpha,
            "noise_ceiling_path": cli.noise_ceiling,
            "diffusion_path": cli.diffusion,
            "ridge_path": cli.ridge,
        },
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    models = []
    if cli.ridge is not None:
        models.append(("Ridge regression", cli.ridge, "#1f77b4", "#00095b"))
    if cli.diffusion is not None:
        models.append(("Diffusion model", cli.diffusion, "#d62728", "#8b0000"))

    # Per-model `all_r` (subject-wise per-voxel r) collected for the paired
    # diffusion − ridge analysis after the scatter loop.
    per_model_all_r = {}

    for label, path, color, line_colour in models:
        enc_r = load_encoding_r(path, label)
        print(f"{label}: {enc_r.shape[0]} pixels", flush=True)
        if len(enc_r) != n_pixels:
            print(
                f"WARNING: {label} has {len(enc_r)} pixels but projected noise ceiling "
                f"has {n_pixels}. Skipping."
            )
            continue

        # Scatter plot — filter NaN voxels
        valid = np.isfinite(nc_r) & np.isfinite(enc_r)
        x = nc_r[valid]
        y = enc_r[valid]

        ax.scatter(
            x, y,
            s=cli.point_size, alpha=cli.alpha, color=color,
            label=f"{label} ({np.sum(valid)} voxels)",
            rasterized=True,
        )

        # Fit and plot trend line
        if len(x) > 2:
            coeffs = np.polyfit(x, y, deg=1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color=line_colour, linewidth=1.5)

            corr = np.corrcoef(x, y)[0, 1]
            print(f"{label}: slope={coeffs[0]:.3f}, r(NC, enc)={corr:.3f}, n={len(x)}")

        # Collect subject-wise r for the diffusion − ridge paired analysis.
        try:
            all_r, subj_names = load_subject_wise_r(path, label)
        except KeyError as e:
            print(f"WARNING: {label} — no per-subject r matrix available: {e}", flush=True)
            continue

        if all_r.shape[1] != n_pixels:
            print(
                f"WARNING: {label} all_r has {all_r.shape[1]} voxels but projected "
                f"noise ceiling has {n_pixels}. Excluding from paired analysis."
            )
            continue

        per_model_all_r[label] = {"all_r": all_r, "subj_names": subj_names}

    # Identity line (perfect encoding = noise ceiling)
    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, upper], [0, upper], "k--", linewidth=0.8, alpha=0.5, label="Identity (y = x)")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)

    ax.set_xlabel("Interparticipant agreement (NC lower bound)")
    ax.set_ylabel("Encoding model performance (group mean r)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()

    ax.set_title(f"Encoding performance vs interparticipant agreement for ROI {cfg.data.roi}", fontsize=12)

    fig.savefig(cli.output, dpi=300, bbox_inches="tight")
    print(f"Saved to {cli.output}")

    # Also log the scatter to wandb so it lives next to the brain maps.
    # Pass the matplotlib Figure directly — wandb.Image cannot read PDFs.
    wandb.log({"scatter/encoding_vs_noise_ceiling": wandb.Image(fig)})
    plt.close(fig)

    # ── Paired subject-wise significance analysis: diffusion − ridge ──
    diff_result = _run_diffusion_minus_ridge_analysis(
        per_model_all_r=per_model_all_r,
        image_shape=image_shape,
        y_coords=y_coords,
        x_coords=x_coords,
        fdr_alpha=cli.fdr_alpha,
        output_dir=Path(cli.output).parent,
    )

    wandb.finish()

    return diff_result


if __name__ == "__main__":
    main()
