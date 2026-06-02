"""Permutation test for the inter-subject noise-ceiling lower bound.

For each voxel, computes an empirical null for the across-subjects one-sample
t-statistic of per-subject leave-one-out (LOO) inter-subject correlation
vs 0, by shuffling stimulus labels independently per subject.

Why this permutation scheme:
  Inter-subject NC at voxel v is the mean across subjects of
      r_S(v) = pearson(Y_S(v), mean_{T != S} Y_T(v))
  over the 515 common test stimuli. Under H0 (no real inter-subject
  reliability), shuffling each subject's stimulus order independently breaks
  the stimulus-level alignment across subjects, so all r_S → 0 and the
  across-subjects t-stat → 0. Permuting subject identity (the naive
  alternative) preserves the per-stimulus consensus and so has near-zero
  power — see CLAUDE.md / earlier-session discussion for the analysis.

Output (vertex space, length = len(roi_indices)):
  - `r_per_subject_real (n_subj, n_voxels)` : real per-subject LOO r.
  - `t_real (n_voxels,)`                     : real across-subjects t-stat.
  - `p_values (n_voxels,)`                   : two-tailed p, from parametric
                                               t-fit to each voxel's null col.
  - `sig_mask (n_voxels,)` (default)         : |t_real| > parametric two-tailed
                                               critical value at alpha.
  - `sig_mask_pct95 (n_voxels,)`             : t_real > empirical per-voxel
                                               95th percentile (one-tailed).
  - `sig_mask_fwer (n_voxels,)`              : |t_real| > 95th percentile of
                                               per-perm max |t| (two-tailed
                                               FWER via max-stat).
  - `upper_crit`, `lower_crit`, `df_per_voxel`: parametric t-fit per voxel.
  - `empirical_upper_crit`, `empirical_lower_crit`: empirical 97.5 / 2.5
                                               percentiles of null per voxel.
  - `pct95_per_voxel`, `fwer_threshold`      : alternate-mask thresholds.
  - `roi_indices`                            : vertex indices used so
                                               `signal_to_2d` can re-grid.

Run via SLURM (host has no Python deps):
    sbatch noise_ceiling_intersubject_perm.sh
"""

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion_brain.utils.setup import (
    load_config_from_yaml,
    parse_value,
    set_nested_attr,
)
from diffusion_brain.utils.fmri_behav_data_utils import get_roi_mask, signal_to_2d
from diffusion_brain.scripts.noise_ceiling_intersubject import (
    ALL_SUBJECTS,
    load_averaged_betas_515,
)


def loo_r_all_subjects(stack):
    """Vectorised per-subject LOO Pearson r.

    Parameters
    ----------
    stack : (n_subj, n_images, n_voxels) float

    Returns
    -------
    r_per_subject : (n_subj, n_voxels)
        ``r_per_subject[i, v] = pearson(stack[i, :, v], mean_{j != i} stack[j, :, v])``.
    """
    n_subj = stack.shape[0]
    n_images = stack.shape[1]
    total_sum = stack.sum(axis=0)                              # (n_images, n_voxels)
    others = (total_sum[None, :, :] - stack) / (n_subj - 1)    # (n_subj, n_images, n_voxels)

    a_centred = stack - stack.mean(axis=1, keepdims=True)
    b_centred = others - others.mean(axis=1, keepdims=True)
    cov = (a_centred * b_centred).sum(axis=1) / (n_images - 1)
    std_a = a_centred.std(axis=1, ddof=1)
    std_b = b_centred.std(axis=1, ddof=1)
    denom = std_a * std_b
    denom_safe = np.where(denom == 0, 1.0, denom)
    return (cov / denom_safe).astype(np.float64)


def t_across_subjects(r_per_subject):
    """One-sample t-stat across subjects vs popmean=0, per voxel.

    Returns (n_voxels,) array. Voxels where the across-subjects std is 0
    receive NaN (no variance → t undefined).
    """
    n_subj = r_per_subject.shape[0]
    mean = r_per_subject.mean(axis=0)
    std = r_per_subject.std(axis=0, ddof=1)
    t = np.full_like(mean, np.nan, dtype=np.float64)
    valid = std > 0
    t[valid] = mean[valid] / (std[valid] / np.sqrt(n_subj))
    return t


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Permutation test for inter-subject NC lower bound",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config (uses data section for ROI/paths)")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--jobid", type=str, default=None)
    parser.add_argument("--n-perms", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Two-tailed alpha (default 0.05)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subjects to include (default: all 8)")
    args_cli = parser.parse_args()

    args = load_config_from_yaml(args_cli.config)
    for ov in args_cli.override:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        set_nested_attr(args, k, parse_value(v))
        print(f"Override: {k} = {parse_value(v)}", flush=True)

    subjects = list(args_cli.subjects or ALL_SUBJECTS)
    roi_label = f"{args.data.roi_file}_{args.data.roi}"

    run_id = args_cli.jobid or f"nc-intersubj-perm-{roi_label}"
    wandb.init(
        id=run_id,
        name=run_id,
        project="noise-ceiling-fmri",
        config={
            "subjects": subjects,
            "roi": args.data.roi,
            "roi_file": args.data.roi_file,
            "n_perms": args_cli.n_perms,
            "alpha": args_cli.alpha,
            "type": "intersubject-permutation",
        },
    )

    print(f"\n{'=' * 60}", flush=True)
    print(f"Inter-subject NC permutation test", flush=True)
    print(f"  ROI:    {args.data.roi_file} / {args.data.roi}", flush=True)
    print(f"  Subj:   {subjects}", flush=True)
    print(f"  Perms:  {args_cli.n_perms}  alpha: {args_cli.alpha}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # ── Load averaged betas for the 515 common stimuli, ROI-masked ──
    roi_indices = get_roi_mask(args)
    print(f"ROI mask: {len(roi_indices)} voxels", flush=True)

    behav_root = args.data.behav_data_root
    averaged_root = "/data/datapool3/datasets/nsd_betas_condavg/"

    subj_stack = []
    for subj in subjects:
        print(f"  Loading {subj} ...", end=" ", flush=True)
        betas = load_averaged_betas_515(subj, roi_indices, behav_root, averaged_root)
        print(f"shape: {betas.shape}", flush=True)
        subj_stack.append(betas)
    # (n_subj, n_images, n_voxels)
    stack = np.array(subj_stack, dtype=np.float32)
    n_subj, n_images, n_voxels = stack.shape
    print(f"  Stack shape: {stack.shape}", flush=True)

    # ── Real statistic ──
    print("\nComputing REAL per-subject LOO r and t-stat ...", flush=True)
    r_per_subject_real = loo_r_all_subjects(stack)              # (n_subj, n_voxels)
    t_real = t_across_subjects(r_per_subject_real)              # (n_voxels,)
    mean_r_real = r_per_subject_real.mean(axis=0)
    print(f"  mean r_lower (real) = {float(np.nanmean(mean_r_real)):+.4f}", flush=True)
    print(f"  mean |t_real|       = {float(np.nanmean(np.abs(t_real))):+.4f}", flush=True)

    # ── Permutation loop ──
    print(f"\nRunning {args_cli.n_perms} permutations "
          f"(shuffling stimulus labels per subject independently)...", flush=True)
    rng = np.random.default_rng(args_cli.seed)
    null_dist = np.zeros((args_cli.n_perms, n_voxels), dtype=np.float32)
    permuted = np.empty_like(stack)

    log_every = max(args_cli.n_perms // 20, 1)
    for p in range(args_cli.n_perms):
        for s in range(n_subj):
            perm = rng.permutation(n_images)
            permuted[s] = stack[s, perm]
        r_perm = loo_r_all_subjects(permuted)
        null_dist[p] = t_across_subjects(r_perm).astype(np.float32)
        if (p + 1) % log_every == 0:
            null_so_far = null_dist[: p + 1]
            print(
                f"  perm {p + 1:5d}/{args_cli.n_perms}  "
                f"mean null |t| = {float(np.nanmean(np.abs(null_so_far))):.4f}",
                flush=True,
            )

    # ── Per-voxel empirical critical values (two-tailed alpha) ──
    half_alpha = args_cli.alpha / 2.0
    empirical_upper_crit = np.nanpercentile(
        null_dist, 100 * (1 - half_alpha), axis=0
    ).astype(np.float64)
    empirical_lower_crit = np.nanpercentile(
        null_dist, 100 * half_alpha, axis=0
    ).astype(np.float64)
    pct95_per_voxel = np.nanpercentile(null_dist, 95.0, axis=0).astype(np.float64)

    # ── Global FWER threshold (max |t| across voxels per perm, 95th percentile) ──
    perm_max_abs = np.nanmax(np.abs(null_dist), axis=1)
    fwer_threshold = float(np.nanpercentile(perm_max_abs, 95.0))
    print(f"\n  global FWER threshold (max |t|, alpha=0.05): {fwer_threshold:.4f}", flush=True)

    # ── Parametric t-distribution fit per voxel (matches within-subject perm) ──
    print(f"  Fitting t-distribution per voxel (two-tailed alpha={args_cli.alpha})...",
          flush=True)
    upper_crit = np.zeros(n_voxels, dtype=np.float64)
    lower_crit = np.zeros(n_voxels, dtype=np.float64)
    p_values = np.ones(n_voxels, dtype=np.float64)
    df_per_voxel = np.zeros(n_voxels, dtype=np.float64)
    for v in range(n_voxels):
        col = null_dist[:, v].astype(np.float64)
        col = col[np.isfinite(col)]
        if col.size < 2:
            df, loc, scale = 1e6, 0.0, 1.0
        else:
            try:
                df, loc, scale = scipy.stats.t.fit(col)
            except Exception:
                df, loc, scale = 1e6, float(col.mean()), float(col.std() or 1.0)
            if not np.isfinite(scale) or scale <= 0:
                scale = max(float(col.std()), 1e-12)
            if not np.isfinite(df) or df <= 0:
                df = 1e6
        df_per_voxel[v] = df
        upper_crit[v] = scipy.stats.t.ppf(1 - half_alpha, df, loc=loc, scale=scale)
        lower_crit[v] = scipy.stats.t.ppf(half_alpha, df, loc=loc, scale=scale)
        if np.isfinite(t_real[v]):
            cdf_real = scipy.stats.t.cdf(t_real[v], df, loc=loc, scale=scale)
            p_values[v] = 2.0 * min(cdf_real, 1.0 - cdf_real)

    # ── Three significance masks (mirror within-subject perm) ──
    sig_mask = (t_real > upper_crit) | (t_real < lower_crit)
    sig_mask_pct95 = t_real > pct95_per_voxel
    sig_mask_fwer = np.abs(t_real) > fwer_threshold

    # Voxels where t_real is NaN (zero across-subjects variance) cannot be
    # called significant under any scheme.
    nan_t = ~np.isfinite(t_real)
    sig_mask[nan_t] = False
    sig_mask_pct95[nan_t] = False
    sig_mask_fwer[nan_t] = False

    n_t = int(sig_mask.sum())
    n_p = int(sig_mask_pct95.sum())
    n_f = int(sig_mask_fwer.sum())
    print(f"\n  Significant voxels:", flush=True)
    print(
        f"    parametric t-fit, two-tailed alpha={args_cli.alpha}: "
        f"{n_t}/{n_voxels} ({100 * n_t / max(n_voxels, 1):.1f}%)",
        flush=True,
    )
    print(
        f"    empirical per-voxel 95th pct (one-tailed):       "
        f"{n_p}/{n_voxels} ({100 * n_p / max(n_voxels, 1):.1f}%)",
        flush=True,
    )
    print(
        f"    FWER max-|t| (two-tailed alpha=0.05):            "
        f"{n_f}/{n_voxels} ({100 * n_f / max(n_voxels, 1):.1f}%)",
        flush=True,
    )

    wandb.log({
        "n_voxels": int(n_voxels),
        "n_sig_param_t": n_t,
        "n_sig_pct95": n_p,
        "n_sig_fwer": n_f,
        "frac_sig_param_t": float(n_t / max(n_voxels, 1)),
        "frac_sig_pct95": float(n_p / max(n_voxels, 1)),
        "frac_sig_fwer": float(n_f / max(n_voxels, 1)),
        "fwer_threshold": fwer_threshold,
        "mean_r_lower_real": float(np.nanmean(mean_r_real)),
        "mean_t_real": float(np.nanmean(t_real)),
    })

    # ── Save (vertex space; `signal_to_2d` at load time re-grids to 2D) ──
    if args_cli.output:
        out_path = args_cli.output
    else:
        out_dir = os.path.join("results", "noise_ceiling")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"noise_ceiling_intersubject_perm_{args.data.roi_file}_{args.data.roi}.npz",
        )

    np.savez(
        out_path,
        # Real statistics
        r_per_subject_real=r_per_subject_real,
        mean_r_lower_real=mean_r_real,
        t_real=t_real,
        # Masks
        sig_mask=sig_mask,
        sig_mask_pct95=sig_mask_pct95,
        sig_mask_fwer=sig_mask_fwer,
        # Parametric t-fit per voxel
        upper_crit=upper_crit,
        lower_crit=lower_crit,
        df_per_voxel=df_per_voxel,
        p_values=p_values,
        # Empirical critical values
        empirical_upper_crit=empirical_upper_crit,
        empirical_lower_crit=empirical_lower_crit,
        pct95_per_voxel=pct95_per_voxel,
        fwer_threshold=fwer_threshold,
        # Metadata (vertex space — re-grid via signal_to_2d at load time)
        roi_indices=roi_indices,
        subjects=np.array(subjects),
        n_perms=args_cli.n_perms,
        alpha=args_cli.alpha,
        seed=args_cli.seed,
        roi=args.data.roi,
        roi_file=args.data.roi_file,
    )
    print(f"\nResults saved to {out_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
