"""Pooled multi-factor linear encoding model across all NSD subjects.

A single RidgeCV is fit on the concatenated training pool of all 8 subjects.
Subject identity enters as an explicit factor in the design matrix:

  - ``additive``    : X = [subject_dummies | ANN]
                      → shared β (ANN→fMRI) + per-subject intercept α_s
  - ``interaction`` : X = [subject_dummies | ANN | subject_dummies ⊗ ANN]
                      → per-subject slope and intercept (still one fit, shared α via RidgeCV)

For each voxel the design is the same, but ``alpha_per_target=True`` lets every
voxel pick its own ridge penalty.

Per-subject and group test-set Pearson r are computed exactly as in
``train_sklearn_group.py`` so the result is comparable.

Run via SLURM, e.g.::

    sbatch 2d_baseline_train/2d_multifactor_no_gpu_run_ann_brain_train.sh \
        $roi $weights $model $roi_file
"""

from sklearn.linear_model import RidgeCV
import torch
import numpy as np
from scipy.stats import ttest_1samp
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
from diffusion_brain.utils.nc_correction import (
    FDR_ALPHA,
    benjamini_hochberg,
    build_brain_map,
    evaluate_nc_corrected_group,
    per_voxel_r,
)
from diffusion_brain.utils.visualise import fmri_to_wandb_image


def load_subject_data(args, subj):
    """Load train/test numpy arrays for a single subject.

    Mirrors ``train_sklearn_group.load_subject_data`` exactly so the pooled
    design here is comparable to the per-subject baseline.
    """
    args.data.subj = subj

    train_nsd_ids, test_nsd_ids, train_indices, test_indices = get_train_test_indices(args)
    fmri_roi_path = ensure_fmri_roi_exists(args)

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

    train_fmri, train_ann = next(iter(torch.utils.data.DataLoader(
        train_ds, batch_size=len(train_ds), shuffle=False)))
    test_fmri, test_ann = next(iter(torch.utils.data.DataLoader(
        test_ds, batch_size=len(test_ds), shuffle=False)))

    return (train_fmri.numpy(), train_ann.numpy(),
            test_fmri.numpy(), test_ann.numpy())


def extract_roi_voxels(fmri_2d, args):
    """Extract active ROI voxel time series from 2D images.

    Locations are identical across subjects (computed from fsaverage
    geometry), so voxel index `i` refers to the same vertex for every subject.
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


def build_design_matrix(ann, subj_idx, n_subjects, mode):
    """Construct the multi-factor design matrix.

    Args:
        ann: (N, F) z-scored ANN activations
        subj_idx: (N,) integer in [0, n_subjects)
        n_subjects: total number of subject levels
        mode: ``additive`` or ``interaction``

    Returns:
        X: (N, D) float32 design matrix.
            additive    : D = n_subjects + F
            interaction : D = n_subjects + F + n_subjects * F
    """
    N, F = ann.shape
    onehot = np.zeros((N, n_subjects), dtype=np.float32)
    onehot[np.arange(N), subj_idx] = 1.0

    if mode == "additive":
        return np.hstack([onehot, ann.astype(np.float32, copy=False)])

    if mode == "interaction":
        # subject × ann interaction block: (N, n_subjects * F)
        # for sample i with subject s, columns [s*F : (s+1)*F] = ann[i], rest = 0
        interaction = np.zeros((N, n_subjects * F), dtype=np.float32)
        rows = np.arange(N)
        for s in range(n_subjects):
            mask = subj_idx == s
            interaction[np.ix_(rows[mask], np.arange(s * F, (s + 1) * F))] = ann[mask]
        return np.hstack([onehot, ann.astype(np.float32, copy=False), interaction])

    raise ValueError(f"Unknown interaction_mode: {mode}")


def main():
    args = parse_args_and_setup_wandb()
    subjects = list(getattr(args.data, "subjects", ALL_SUBJECTS))
    n_subjects = len(subjects)
    is_2d = args.data.is_2d
    mode = getattr(args.model, "interaction_mode", "additive")
    if mode not in {"additive", "interaction"}:
        raise ValueError(f"model.interaction_mode must be 'additive' or 'interaction', got {mode!r}")

    print(f"\nMulti-factor regression: mode={mode}, subjects={subjects}", flush=True)

    train_ann_list, train_voxels_list, train_subj_list = [], [], []
    test_per_subject = {}  # subj -> (test_ann, test_voxels)
    image_shape = None
    locations_roi = None

    for s_idx, subj in enumerate(subjects):
        print(f"\n{'='*60}\nLoading {subj}\n{'='*60}", flush=True)
        train_fmri, train_ann, test_fmri, test_ann = load_subject_data(args, subj)

        if is_2d:
            train_y, loc = extract_roi_voxels(train_fmri, args)
            test_y, _ = extract_roi_voxels(test_fmri, args)
            if image_shape is None:
                image_shape = train_fmri.squeeze(1).shape[1:]
                locations_roi = loc
        else:
            train_y, test_y = train_fmri, test_fmri

        train_ann_list.append(train_ann)
        train_voxels_list.append(train_y)
        train_subj_list.append(np.full(train_ann.shape[0], s_idx, dtype=np.int64))
        test_per_subject[subj] = (test_ann, test_y)

        print(f"  train: ann {train_ann.shape}, fmri {train_y.shape}", flush=True)
        print(f"  test : ann {test_ann.shape}, fmri {test_y.shape}", flush=True)

    train_ann_pool = np.vstack(train_ann_list)
    train_y_pool = np.vstack(train_voxels_list).astype(np.float32, copy=False)
    train_subj_pool = np.concatenate(train_subj_list)

    n_voxels = train_y_pool.shape[1]
    print(f"\nPooled training matrix: ann {train_ann_pool.shape}, "
          f"fmri {train_y_pool.shape}, n_subjects={n_subjects}", flush=True)

    X_train = build_design_matrix(train_ann_pool, train_subj_pool, n_subjects, mode)
    print(f"Design matrix X_train: {X_train.shape} ({mode})", flush=True)

    # Free intermediates that are no longer needed
    del train_ann_list, train_voxels_list, train_subj_list
    del train_ann_pool, train_subj_pool

    alphas = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    print(f"Fitting RidgeCV (alpha_per_target=True, alphas={alphas.tolist()})...", flush=True)
    clf = RidgeCV(alphas=alphas, alpha_per_target=True)
    clf.fit(X_train, train_y_pool)
    del X_train, train_y_pool

    print(f"  selected alpha summary: min={clf.alpha_.min():.3g}, "
          f"median={np.median(clf.alpha_):.3g}, max={clf.alpha_.max():.3g}", flush=True)

    # Per-identity-factor test predictions (one per subject, computed once).
    # Each `pred_per_identity[i]` predicts subject-i's expected response on the
    # 515 shared test stimuli using subject-i's z-scored ANN. The off-diagonal
    # of the confusion matrix below is what's interesting for this baseline:
    # in `additive` mode predictions for different identities differ only by a
    # per-subject intercept — so per-voxel Pearson r (shift-invariant) is
    # identical across rows and the matrix should show ~zero specificity. In
    # `interaction` mode each identity gets its own slope, so rows actually
    # diverge.
    pred_per_identity = {}
    for s_idx, subj in enumerate(subjects):
        test_ann, _ = test_per_subject[subj]
        N_test = test_ann.shape[0]
        subj_idx_test = np.full(N_test, s_idx, dtype=np.int64)
        X_test = build_design_matrix(test_ann, subj_idx_test, n_subjects, mode)
        pred_per_identity[subj] = clf.predict(X_test)

    # Confusion matrix: rows = identity factor i (in design matrix),
    # columns = subject whose true fMRI we compare against.
    confusion = np.full((n_subjects, n_subjects), np.nan, dtype=np.float64)
    all_r = np.zeros((n_subjects, n_voxels), dtype=np.float32)
    per_subject_r_diag = {}
    for i_idx, subj_i in enumerate(subjects):
        pred_i = pred_per_identity[subj_i]
        for j_idx, subj_j in enumerate(subjects):
            _, test_y_j = test_per_subject[subj_j]
            r_vox = per_voxel_r(pred_i, test_y_j)
            confusion[i_idx, j_idx] = float(np.nanmean(r_vox))
            if i_idx == j_idx:
                all_r[i_idx] = r_vox.astype(np.float32)
                per_subject_r_diag[subj_i] = r_vox

        diag_r = all_r[i_idx]
        print(f"{subj_i}: mean r = {np.nanmean(diag_r):.4f}, "
              f"median r = {np.nanmedian(diag_r):.4f}", flush=True)
        wandb.log({f"r_mean/{subj_i}": float(np.nanmean(diag_r))})

    # Print the matrix
    print(f"\n{'=' * 60}", flush=True)
    print(f"Cross-identity multifactor confusion matrix ({mode})", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  rows = identity factor i in design matrix", flush=True)
    print(f"  cols = true fMRI from subject j", flush=True)
    print(f"  cell = mean per-voxel Pearson r over {n_voxels} voxels", flush=True)
    col_header = "             " + "  ".join(f"{n:>8}" for n in subjects)
    print(col_header, flush=True)
    for i_idx, i_name in enumerate(subjects):
        row_vals = "  ".join(f"{confusion[i_idx, j_idx]:+8.4f}" for j_idx in range(n_subjects))
        print(f"  {i_name:>10}  {row_vals}", flush=True)

    diag_vals = np.diag(confusion)
    off_mask = ~np.eye(n_subjects, dtype=bool)
    mean_diag = float(np.nanmean(diag_vals))
    mean_off = float(np.nanmean(confusion[off_mask]))
    specificity = mean_diag - mean_off
    print(f"\n  mean(diagonal)   = {mean_diag:+.4f}", flush=True)
    print(f"  mean(off-diag)   = {mean_off:+.4f}", flush=True)
    print(
        f"  specificity      = {specificity:+.4f}  "
        f"(in '{mode}' mode; additive mode is expected to be ~0)",
        flush=True,
    )
    print(f"{'=' * 60}\n", flush=True)

    confusion_log = {
        "confusion/mean_diagonal": mean_diag,
        "confusion/mean_offdiagonal": mean_off,
        "confusion/specificity": specificity,
    }
    for i_idx, i_name in enumerate(subjects):
        for j_idx, j_name in enumerate(subjects):
            confusion_log[f"confusion/cell/identity_{i_name}_vs_true_{j_name}"] = float(
                confusion[i_idx, j_idx]
            )

    finite = confusion[np.isfinite(confusion)]
    vmax = max(float(np.nanmax(np.abs(finite))) if finite.size > 0 else 1.0, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(1.1 * n_subjects + 2, 1.1 * n_subjects + 1.5))
    im = ax.imshow(confusion, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(n_subjects))
    ax.set_yticks(range(n_subjects))
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_yticklabels(subjects)
    ax.set_xlabel("true subject (columns)")
    ax.set_ylabel("identity factor in design matrix (rows)")
    ax.set_title(
        f"Multifactor Ridge confusion ({mode})\n"
        f"diag={mean_diag:+.4f}, off={mean_off:+.4f}, spec={specificity:+.4f}"
    )
    for i_idx in range(n_subjects):
        for j_idx in range(n_subjects):
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
    confusion_log["confusion/matrix_heatmap"] = wandb.Image(fig)
    plt.close(fig)
    wandb.log(confusion_log)

    # ── Per-subject sample previews (first 3 generated + true 2D maps) ──
    # Mirrors generate.py's per-subject visualisation: for each subject log the
    # first three test stimuli as 2D flatmap images so generated/true pairs can
    # be eyeballed for spatial fidelity. 2D-only — locations_roi is needed to
    # project the (n_voxels,) prediction back onto the ROI grid.
    if is_2d and image_shape is not None:
        y_coords, x_coords = locations_roi[0], locations_roi[1]
        n_preview = 3
        for subj in subjects:
            pred_subj = pred_per_identity[subj]              # (n_stim, n_voxels)
            true_subj = test_per_subject[subj][1]            # (n_stim, n_voxels)
            n_show = min(n_preview, pred_subj.shape[0])
            sample_log = {}
            for i in range(n_show):
                gen_2d = build_brain_map(pred_subj[i], image_shape, y_coords, x_coords)
                true_2d = build_brain_map(true_subj[i], image_shape, y_coords, x_coords)
                sample_log[f"samples/{subj}/generated_{i}"] = fmri_to_wandb_image(
                    gen_2d, title=f"{subj} generated sample {i} ({mode})", robust=True,
                )
                sample_log[f"samples/{subj}/true_{i}"] = fmri_to_wandb_image(
                    true_2d, title=f"{subj} true fMRI sample {i}", robust=True,
                )
            wandb.log(sample_log)

    # ── Group stats + noise-ceiling correction (mirrors confusion.py path) ──
    nc_results = evaluate_nc_corrected_group(
        per_subject_r_diag=per_subject_r_diag,
        subj_names=list(subjects),
        args=args,
        locations_roi=locations_roi,
        image_shape=image_shape,
        is_2d=is_2d,
        label=f"Multifactor Ridge ({mode})",
    )

    # Fallback group stats when NC helper bails out (1D / no locations).
    if nc_results is None:
        group_mean_r = np.nanmean(all_r, axis=0)
        t_stats, p_values = ttest_1samp(all_r, popmean=0.0, axis=0, nan_policy="omit")
        p_values = np.asarray(p_values, dtype=np.float64)
        p_values[np.isnan(p_values)] = 1.0
        reject, p_corrected = benjamini_hochberg(p_values, alpha=FDR_ALPHA)
        n_sig = int(reject.sum())
        print(f"\nSignificant voxels (BH-FDR q<{FDR_ALPHA}): {n_sig}/{n_voxels} "
              f"({100 * n_sig / n_voxels:.1f}%)", flush=True)
        print(f"Group mean r (all voxels): {np.nanmean(group_mean_r):.4f}", flush=True)
        if n_sig > 0:
            print(f"Group mean r (significant only): {np.nanmean(group_mean_r[reject]):.4f}",
                  flush=True)
        wandb.log({
            "group/mean_r_all_voxels": float(np.nanmean(group_mean_r)),
            "group/mean_r_significant": float(np.nanmean(group_mean_r[reject])) if n_sig > 0 else 0.0,
            "group/n_significant_voxels": n_sig,
            "group/n_total_voxels": int(n_voxels),
            "group/frac_significant": float(n_sig / n_voxels),
            "config/interaction_mode": mode,
        })
    else:
        group_mean_r = nc_results["group_mean_r"]
        t_stats = nc_results["t_stats"]
        p_values = nc_results["p_values"]
        p_corrected = nc_results["p_corrected"]
        reject = nc_results["reject"]
        n_sig = int(reject.sum())
        if n_sig > 0:
            wandb.log({
                "group/mean_r_significant": float(np.nanmean(group_mean_r[reject])),
            })
        wandb.log({"config/interaction_mode": mode})

    # Visualisation (2D only) — unthresholded, FDR-thresholded, significance mask
    if is_2d and image_shape is not None:
        y_coords, x_coords = locations_roi[0], locations_roi[1]

        for tag, values, title in [
            ("group/r_map_unthresholded", group_mean_r,
             f"Group r ({mode}, N={n_subjects}, all voxels)"),
            ("group/r_map_fdr_thresholded",
             np.where(reject, group_mean_r, 0.0),
             f"Group r ({mode}, N={n_subjects}, BH-FDR q<{FDR_ALPHA}, "
             f"{n_sig}/{n_voxels} sig)"),
        ]:
            r_img = np.zeros(image_shape, dtype=np.float32)
            r_img[y_coords, x_coords] = values
            abs_max = max(abs(r_img.min()), abs(r_img.max()), 1e-8)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(r_img, cmap="RdBu_r", origin="lower", norm=norm)
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
            wandb.log({tag: wandb.Image(fig)})
            plt.close(fig)

        sig_img = np.zeros(image_shape, dtype=np.float32)
        sig_img[y_coords, x_coords] = reject.astype(float)
        fig_sig, ax_sig = plt.subplots(figsize=(8, 6))
        ax_sig.imshow(sig_img, cmap="Greys", origin="lower", vmin=0, vmax=1)
        ax_sig.set_title(
            f"Significant voxels ({mode}, BH-FDR q<{FDR_ALPHA}, "
            f"{n_sig}/{n_voxels} sig, {100.0 * n_sig / max(n_voxels, 1):.1f}%)"
        )
        wandb.log({"group/significance_mask": wandb.Image(fig_sig)})
        plt.close(fig_sig)

    output_dir = os.path.join(
        getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
        args.jobid,
    )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"multifactor_ridge_results_{mode}.npz")

    # Stack per-subject test predictions into one (n_subjects, n_test_stim, n_voxels)
    # array. Row order matches `subjects`; column j corresponds to NSD image
    # `test_nsd_ids_for_save[j]`. The comparison below looks up by NSD id, not
    # by column position, so an external ordering change can't silently
    # produce wrong pairings.
    pred_per_identity_arr = np.stack(
        [pred_per_identity[s] for s in subjects], axis=0
    ).astype(np.float32)

    # Capture the 515 test NSD ids. In the averaged variant these come from a
    # shared `common_515_indices.npy` file and are subject-independent — verify
    # this rather than trusting it implicitly, so a future ordering change
    # surfaces here instead of producing silent misalignment downstream.
    test_nsd_ids_per_subject = []
    for s in subjects:
        args.data.subj = s
        _, t_ids, _, _ = get_train_test_indices(args)
        test_nsd_ids_per_subject.append(np.asarray(t_ids))
    test_nsd_ids_for_save = test_nsd_ids_per_subject[0]
    for s_idx, t_ids in enumerate(test_nsd_ids_per_subject[1:], start=1):
        if not np.array_equal(test_nsd_ids_for_save, t_ids):
            raise RuntimeError(
                f"Test NSD-id ordering differs between subjects "
                f"{subjects[0]!r} and {subjects[s_idx]!r}; pred_per_identity "
                "rows would not be aligned across subjects. Refusing to save."
            )

    save_dict = dict(
        all_r=all_r,
        group_mean_r=group_mean_r,
        t_stats=t_stats,
        p_values=p_values,
        p_corrected=p_corrected,
        reject=reject,
        subjects=np.array(subjects),
        alphas_per_voxel=clf.alpha_,
        interaction_mode=mode,
        confusion=confusion,
        confusion_mean_diagonal=mean_diag,
        confusion_mean_offdiagonal=mean_off,
        confusion_specificity=specificity,
        pred_per_identity=pred_per_identity_arr,
        test_nsd_ids=np.asarray(test_nsd_ids_for_save),
        roi=np.array(args.data.roi),
        roi_file=np.array(args.data.roi_file),
    )
    if is_2d and locations_roi is not None:
        save_dict["y_coords"] = locations_roi[0]
        save_dict["x_coords"] = locations_roi[1]
        save_dict["image_shape"] = np.array(image_shape)
    if nc_results is not None:
        # NC-specific fields only — the t-test/BH-FDR fields produced by the
        # helper are already saved above (computed identically from `all_r`).
        for k in (
            "nc_lower_bound_intersubj",
            "group_r_corrected",
            "group_apply_mask",
            "per_subject_r_corrected_intersubj",
            "per_subject_r_corrected_intersubj_subjects",
        ):
            if k in nc_results:
                save_dict[k] = nc_results[k]
    np.savez(out_path, **save_dict)
    print(f"\nResults saved to {out_path}", flush=True)

    # Optional: build 3-panel comparison GIFs against a diffusion generation run.
    # Looks for trajectory_inputs_*.npz under args.validation.diffusion_outputs_dir.
    # Lookup is by NSD image id (not positional index) so this is robust to any
    # future change in test-set ordering on either side.
    diffusion_dir = getattr(args.validation, "diffusion_outputs_dir", None)
    if diffusion_dir:
        _render_diffusion_vs_ridge_gifs(
            diffusion_dir=diffusion_dir,
            output_dir=output_dir,
            pred_per_identity_arr=pred_per_identity_arr,
            subjects=subjects,
            test_nsd_ids_ridge=test_nsd_ids_for_save,
            image_shape=image_shape,
            locations_roi=locations_roi,
            is_2d=is_2d,
            interaction_mode=mode,
        )


def _render_diffusion_vs_ridge_gifs(
    diffusion_dir,
    output_dir,
    pred_per_identity_arr,
    subjects,
    test_nsd_ids_ridge,
    image_shape,
    locations_roi,
    is_2d,
    interaction_mode,
):
    """Render `[diffusion trajectory | Ridge | ground truth]` GIFs by looking
    up Ridge predictions by NSD image id."""
    import glob
    import re
    from io import BytesIO
    from PIL import Image

    if not is_2d or locations_roi is None:
        print("[compare-gif] skipping: only supported for 2D + ROI locations.", flush=True)
        return

    inputs_files = sorted(glob.glob(os.path.join(str(diffusion_dir), "trajectory_inputs_*.npz")))
    if not inputs_files:
        print(f"[compare-gif] no trajectory_inputs_*.npz under {diffusion_dir}", flush=True)
        return

    y_coords, x_coords = locations_roi[0], locations_roi[1]
    ridge_nsd_ids = np.asarray(test_nsd_ids_ridge)
    ridge_nsd_to_pos = {int(nsd_id): i for i, nsd_id in enumerate(ridge_nsd_ids)}
    out_gif_dir = os.path.join(str(output_dir), "diffusion_vs_ridge_gifs")
    os.makedirs(out_gif_dir, exist_ok=True)

    for inputs_path in inputs_files:
        print(f"[compare-gif] processing {inputs_path}", flush=True)
        d = np.load(inputs_path, allow_pickle=True)

        # Mandatory fields for NSD-id-based alignment
        if "selected_nsd_ids" not in d.files:
            print(
                f"[compare-gif] {os.path.basename(inputs_path)} has no "
                "'selected_nsd_ids' — produced by an older generate.py. Skipping "
                "(re-run generation with the patched code to enable comparison).",
                flush=True,
            )
            continue

        traj_t = d["trajectory_t"]
        traj_snaps = d["trajectory_snapshots"]  # (T, B, ...) possibly with channel
        sel_names = [str(n) for n in d["selected_subject_names"]]
        sel_nsd_ids = d["selected_nsd_ids"]
        sel_true = d["selected_true_fmri"]
        if sel_true.ndim == 4:
            sel_true = sel_true.squeeze(1)

        # Defensive: if the diffusion-side npz also recorded its own test NSD
        # ordering, assert it matches the Ridge-side ordering element-wise.
        if "test_nsd_ids" in d.files:
            other = np.asarray(d["test_nsd_ids"])
            if not np.array_equal(other, ridge_nsd_ids):
                print(
                    f"[compare-gif] WARNING: test_nsd_ids in {os.path.basename(inputs_path)} "
                    f"differs from Ridge-side ordering; using per-id lookup anyway.",
                    flush=True,
                )

        tag = re.sub(r"^trajectory_inputs_", "", os.path.splitext(os.path.basename(inputs_path))[0])

        for slot, subj_name in enumerate(sel_names):
            try:
                subj_pos = subjects.index(subj_name)
            except ValueError:
                print(f"[compare-gif] subject {subj_name!r} not in Ridge subjects list; skipping slot.", flush=True)
                continue
            nsd_id = int(sel_nsd_ids[slot])
            ridge_col = ridge_nsd_to_pos.get(nsd_id)
            if ridge_col is None:
                print(
                    f"[compare-gif] NSD id {nsd_id} (subj {subj_name}) not in Ridge test set; skipping slot.",
                    flush=True,
                )
                continue

            ridge_voxels = pred_per_identity_arr[subj_pos, ridge_col]
            ridge_img = np.zeros(image_shape, dtype=np.float32)
            ridge_img[y_coords, x_coords] = np.nan_to_num(ridge_voxels, nan=0.0)
            ridge_vmax = max(abs(float(ridge_img.min())), abs(float(ridge_img.max())), 1e-8)

            gt_img = sel_true[slot]
            gt_vmax = max(abs(float(gt_img.min())), abs(float(gt_img.max())), 1e-8)

            frames = []
            for t_idx, t_val in enumerate(traj_t):
                snap = traj_snaps[t_idx][slot]
                if snap.ndim == 3:
                    snap = snap[0]
                gen_vmax = max(abs(float(snap.min())), abs(float(snap.max())), 1e-8)

                fig, (ax_g, ax_r, ax_t) = plt.subplots(1, 3, figsize=(12, 4))
                ax_g.imshow(
                    snap, cmap="RdBu_r", origin="lower",
                    norm=TwoSlopeNorm(vmin=-gen_vmax, vcenter=0.0, vmax=gen_vmax),
                )
                ax_g.set_title(f"{subj_name} diffusion, t={float(t_val):.3f}")
                ax_g.set_xticks([]); ax_g.set_yticks([])
                ax_r.imshow(
                    ridge_img, cmap="RdBu_r", origin="lower",
                    norm=TwoSlopeNorm(vmin=-ridge_vmax, vcenter=0.0, vmax=ridge_vmax),
                )
                ax_r.set_title(f"{subj_name} Ridge ({interaction_mode}) — NSD {nsd_id}")
                ax_r.set_xticks([]); ax_r.set_yticks([])
                ax_t.imshow(
                    gt_img, cmap="RdBu_r", origin="lower",
                    norm=TwoSlopeNorm(vmin=-gt_vmax, vcenter=0.0, vmax=gt_vmax),
                )
                ax_t.set_title(f"{subj_name} ground truth")
                ax_t.set_xticks([]); ax_t.set_yticks([])
                fig.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                frames.append(Image.open(buf).convert("RGB"))

            gif_path = os.path.join(
                out_gif_dir,
                f"compare_{subj_name}_{tag}_nsd{nsd_id}.gif",
            )
            durations = [300] * (len(frames) - 1) + [10000]
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:],
                duration=durations, loop=1, disposal=2, optimize=False,
            )
            print(f"[compare-gif] wrote {gif_path}", flush=True)


if __name__ == "__main__":
    main()
