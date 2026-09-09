import torch
import os
import re
import wandb
import numpy as np
from scipy.stats import pearsonr, ttest_1samp

from diffusion_brain.utils.diffusivity import generate_samples, get_diffusion
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results, pyplot_brain, fmri_to_wandb_image

N_PREVIEW_PER_SUBJECT = 3
from diffusion_brain.models import set_model, ANNTokenizer
from diffusion_brain.models.autoencoder import LinearAutoencoder, get_linear_autoencoder
from diffusion_brain.data_utils import get_dataloader
from diffusion_brain.utils.fmri_behav_data_utils import signal_to_2d, ALL_SUBJECTS
from diffusion_brain.utils.grad_updaters import EMAModel
from diffusion_brain.utils.nc_correction import (
    FDR_ALPHA,
    NC_MASK_KEY_DEFAULT,
    NC_MASKS_PATH_DEFAULT,
    benjamini_hochberg,
    build_brain_map,
    correct_r_by_nc_with_mask,
    get_intersubject_sig_mask_aligned,
    get_nc_perm_aligned,
    nc_colorbar_vmax,
    per_voxel_r,
)


def _load_subject_locations_2d(args, subj):
    """Load (y_coords, x_coords) for a subject's ROI in the 2D flatmap grid."""
    locations_load_path = os.path.join(
        args.data.roi_defs_dir,
        f"roi_preselected_extended_2d_images_res_{args.data.grid_resolution_2d}",
        f"{args.data.roi_file}",
        f"{subj}_{args.data.roi}.npz",
    )
    locations_roi = np.load(locations_load_path, allow_pickle=True)["locations"]
    return locations_roi[0], locations_roi[1]  # y_coords, x_coords


def _load_noise_ceiling_2d_pixels(args, y_coords, x_coords, mode="intersubject", subj=None):
    """Load noise ceiling lower bound projected to 2D pixel space.

    Parameters
    ----------
    mode : {"intersubject", "per_subject"}
        - ``"intersubject"`` (default, group-level evaluation): loads
          ``noise_ceiling_intersubject_{roi_file}_{roi}.npz`` (field
          ``avg_r_lower``). Falls back to the per-subject file for ``subj``
          if the intersubject file is missing.
        - ``"per_subject"`` (single-subject and per-subject-in-multisubject):
          loads ``noise_ceiling_{subj}_{roi_file}_{roi}.npz`` (field
          ``nc_lower_avg_clamped``).
    subj : str, optional
        Subject ID for the per-subject NC file. Defaults to ``args.data.subj``.

    Returns (nc_pixels, source_label) or (None, None) if the file is missing.
    ``nc_pixels`` has shape ``(n_pixels,)`` aligned with ``(y_coords, x_coords)``.
    """
    nc_dir = os.path.join("results", "noise_ceiling")
    roi_file = args.data.roi_file
    roi = args.data.roi
    if subj is None:
        subj = getattr(args.data, "subj", "subj01")

    intersubj_path = os.path.join(nc_dir, f"noise_ceiling_intersubject_{roi_file}_{roi}.npz")
    persubj_path = os.path.join(nc_dir, f"noise_ceiling_{subj}_{roi_file}_{roi}.npz")

    if mode == "intersubject":
        # Prefer intersubject; fall back to per-subject if missing.
        if os.path.isfile(intersubj_path):
            nc_data = np.load(intersubj_path, allow_pickle=True)
            nc_per_vertex = nc_data["avg_r_lower"]
            source_label = "intersubject"
        elif os.path.isfile(persubj_path):
            nc_data = np.load(persubj_path, allow_pickle=True)
            nc_per_vertex = nc_data["nc_lower_avg_clamped"]
            source_label = f"per-subject ({subj}) [intersubject fallback]"
        else:
            return None, None
    elif mode == "per_subject":
        if not os.path.isfile(persubj_path):
            return None, None
        nc_data = np.load(persubj_path, allow_pickle=True)
        nc_per_vertex = nc_data["nc_lower_avg_clamped"]
        source_label = f"per-subject ({subj})"
    else:
        raise ValueError(f"Unknown noise ceiling mode: {mode!r}")

    # Project vertex-space NC to 2D flatmap pixels via signal_to_2d
    nc_2d, _ = signal_to_2d(args, one_signal_to_transform=nc_per_vertex)
    nc_2d = np.squeeze(nc_2d)  # (H, W)
    nc_pixels = nc_2d[y_coords, x_coords]  # (n_pixels,)
    return nc_pixels, source_label


def _evaluate_per_subject_and_group(
    generated_samples_per_model,
    true_fmri,
    subject_ids,
    args,
    step_num,
    model_name,
):
    """Compute per-subject per-voxel r, print neatly, and log group stats.

    Returns (per_subject_mean_r, group_mean_r, n_significant) for the caller,
    or (None, None, None) when group stacking is not possible.
    """
    # Convert everything to numpy for downstream scipy calls
    gen_np = generated_samples_per_model.cpu().numpy() if isinstance(
        generated_samples_per_model, torch.Tensor) else np.asarray(generated_samples_per_model)
    true_np = true_fmri.cpu().numpy() if isinstance(true_fmri, torch.Tensor) else np.asarray(true_fmri)
    sid_np = subject_ids.cpu().numpy() if isinstance(subject_ids, torch.Tensor) else np.asarray(subject_ids)

    unique_ids = sorted(np.unique(sid_np).tolist())
    header = f"Per-subject generation r-scores ({model_name})"
    print(f"\n{'=' * 60}\n{header}\n{'=' * 60}", flush=True)

    # Gather per-subject per-voxel r arrays. Keep track of voxel counts so we
    # can decide whether group stacking (t-test + BH-FDR) is feasible.
    per_subject_r = []         # list of (n_voxels_subj,) arrays
    per_subject_mean = {}      # subj_idx -> mean r (for quick summary)
    per_subject_median = {}
    per_subject_previews = {}  # subj_name -> {"generated_i": wandb.Image, "true_i": wandb.Image}

    # Grid geometry captured on the first 2D subject iteration; reused for the
    # group-level flatmaps after the loop. Per CLAUDE.md, `locations` arrays are
    # identical across all 8 per-subject ROI cache files, so taking them from any
    # single subject is safe.
    image_shape_ref = None
    y_coords_ref = None
    x_coords_ref = None

    for subj_idx in unique_ids:
        mask = sid_np == subj_idx
        gen_s = gen_np[mask]
        true_s = true_np[mask]
        subj_name = ALL_SUBJECTS[subj_idx] if 0 <= subj_idx < len(ALL_SUBJECTS) else f"subj_id_{subj_idx}"

        # Keep a copy of the raw 2D flatmap slices before we collapse them
        # down to voxel time series — we need the (H, W) grid to render
        # wandb preview images below.
        if args.data.is_2d:
            gen_flat = gen_s.squeeze(1) if gen_s.ndim == 4 else gen_s  # (N, H, W)
            true_flat = true_s.squeeze(1) if true_s.ndim == 4 else true_s
            # Extract the subject's ROI voxel time series from the 2D flatmap.
            y_coords, x_coords = _load_subject_locations_2d(args, subj_name)
            gen_vox = gen_flat[:, y_coords, x_coords]
            true_vox = true_flat[:, y_coords, x_coords]
        else:
            # 1D: samples are already (N, n_voxels) or (N, 1, n_voxels)
            gen_flat = None
            true_flat = None
            if gen_s.ndim == 3:
                gen_s = gen_s.squeeze(1)
            if true_s.ndim == 3:
                true_s = true_s.squeeze(1)
            gen_vox = gen_s
            true_vox = true_s

        r_per_voxel = per_voxel_r(gen_vox, true_vox)

        n_img, n_vox = gen_vox.shape
        mean_r = float(np.nanmean(r_per_voxel))
        median_r = float(np.nanmedian(r_per_voxel))
        per_subject_mean[subj_idx] = mean_r
        per_subject_median[subj_idx] = median_r
        per_subject_r.append(r_per_voxel)

        print(
            f"  {subj_name}: N={n_img}, n_voxels={n_vox}, "
            f"mean r = {mean_r:+.4f}, median r = {median_r:+.4f}",
            flush=True,
        )

        # Build per-subject preview images (first N_PREVIEW_PER_SUBJECT samples
        # of this subject's slice). For 2D we log the full flatmap as a single
        # *list* of wandb.Image objects per row — wandb renders a list under
        # one key as one media panel with all images side-by-side, which keeps
        # the 3 generated and 3 true previews on one screen. For 1D we skip the
        # preview since there is no natural image to render here.
        subj_preview = {}
        if args.data.is_2d and gen_flat is not None:
            n_preview = min(N_PREVIEW_PER_SUBJECT, gen_flat.shape[0])
            generated_imgs = [
                fmri_to_wandb_image(gen_flat[i], title=f"{subj_name} generated {i}")
                for i in range(n_preview)
            ]
            true_imgs = [
                fmri_to_wandb_image(true_flat[i], title=f"{subj_name} true {i}")
                for i in range(n_preview)
            ]
            # Per-subject per-voxel r-score brain map — symmetric to the pooled
            # r_image logged by `visualise_and_save_results`, but restricted to
            # this subject's test slice so subjects can be compared visually.
            # NaN voxels (zero variance) are filled with 0 for rendering.
            subj_image_shape = gen_flat.shape[1:]
            r_img_subj = np.zeros(subj_image_shape, dtype=np.float64)
            r_img_subj[y_coords, x_coords] = np.nan_to_num(r_per_voxel, nan=0.0)
            subj_r_image = fmri_to_wandb_image(
                r_img_subj, title=f"{subj_name} per-voxel r ({model_name})"
            )
            # Both keys live under `per_subject/{subj_name}/` so they share a
            # wandb panel section with the per-subject scalars below.
            subj_preview[f"per_subject/{subj_name}/generated"] = generated_imgs
            subj_preview[f"per_subject/{subj_name}/true"] = true_imgs
            subj_preview[f"per_subject/{subj_name}/r_image"] = subj_r_image
            per_subject_previews[subj_name] = subj_preview

            # Capture shared grid geometry once for the group-level flatmaps.
            if image_shape_ref is None:
                image_shape_ref = subj_image_shape
                y_coords_ref = y_coords
                x_coords_ref = x_coords

        # Per-subject scalars share the same `per_subject/{subj_name}/` prefix
        # as the image keys above, so wandb groups them all into one section
        # per subject (true, generated, mean_r, median_r on one screen).
        log_payload = {
            f"per_subject/{subj_name}/mean_r": mean_r,
            f"per_subject/{subj_name}/median_r": median_r,
            "model_step": step_num,
        }
        log_payload.update(subj_preview)
        wandb.log(log_payload)

    # ── Noise-ceiling correction ──
    # Per-subject correction uses each subject's *within-subject* NC file,
    # because the per-subject raw r reflects within-subject prediction quality.
    # The group-level correction (later) uses the *intersubject* NC, since the
    # multi-subject model captures shared cross-subject variance.
    nc_intersubj_pixels = None
    nc_intersubj_source = None
    per_subject_r_corrected_intersubj = {}
    per_subject_r_corrected_within = {}
    intersubj_sig_mask = None
    intersubj_sig_mask_source = None
    # Shared colorbar bound for the intersubject family (per-subject intersubject
    # maps + the group map). Set once after the per-subject loop, reused for the
    # group map so all intersubject corrections render on one scale.
    inter_vmax = None
    if args.data.is_2d and y_coords_ref is not None:
        nc_intersubj_pixels, nc_intersubj_source = _load_noise_ceiling_2d_pixels(
            args, y_coords_ref, x_coords_ref, mode="intersubject",
        )
        # Per-subject intersubject sig mask (applied ONLY to per-subject
        # intersubject corrections; the group correction stays gated on BH-FDR).
        intersubj_mask_key = getattr(
            args.validation, "nc_intersubj_mask_key", "sig_mask",
        )
        intersubj_sig_mask, intersubj_sig_mask_source = get_intersubject_sig_mask_aligned(
            args.data.roi_file, args.data.roi, y_coords_ref, x_coords_ref, args,
            mask_key=intersubj_mask_key,
        )

    if args.data.is_2d and y_coords_ref is not None:
        print(f"\n{'=' * 60}", flush=True)
        print("Per-subject noise-ceiling-corrected r-scores (within-subject NC)", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Pass 1: compute every subject's within-subject r/NC and stage its
        # wandb payload (scalars + NC-lower-bound / sig-mask images). The r/NC
        # map itself is deferred to pass 2 so all subjects can share one
        # colorbar scale (one bound per correction type).
        within_pending = []  # list of (nc_log_dict, r_corrected, subj_name)
        for subj_idx, r_arr in zip(unique_ids, per_subject_r):
            subj_name = (
                ALL_SUBJECTS[subj_idx]
                if 0 <= subj_idx < len(ALL_SUBJECTS)
                else f"subj_id_{subj_idx}"
            )

            # Prefer the within-subject permutation NC + significance mask
            # from the aggregated pkl. Fall back to the legacy LOO file (no
            # mask) when the pkl doesn't contain this (subj, roi) entry.
            nc_subj_pixels, sig_mask_subj, nc_subj_source = get_nc_perm_aligned(
                subj_name, args.data.roi_file, args.data.roi,
                y_coords_ref, x_coords_ref,
                masks_path=getattr(args.data, "nc_masks_path", NC_MASKS_PATH_DEFAULT),
                mask_key=getattr(args.validation, "nc_mask_key", NC_MASK_KEY_DEFAULT),
            )
            using_perm = nc_subj_pixels is not None
            if not using_perm:
                nc_subj_pixels, nc_subj_source = _load_noise_ceiling_2d_pixels(
                    args, y_coords_ref, x_coords_ref,
                    mode="per_subject", subj=subj_name,
                )
                sig_mask_subj = np.ones_like(r_arr, dtype=bool)
            if nc_subj_pixels is None:
                print(
                    f"  {subj_name}: per-subject NC file missing — skipping "
                    "correction for this subject.",
                    flush=True,
                )
                continue

            r_corrected, apply_mask = correct_r_by_nc_with_mask(
                r_arr, nc_subj_pixels, sig_mask_subj,
            )
            per_subject_r_corrected_within[subj_name] = r_corrected
            n_apply = int(apply_mask.sum())
            mean_rc = float(np.nanmean(r_corrected))
            median_rc = float(np.nanmedian(r_corrected))

            print(
                f"  {subj_name} ({nc_subj_source}): mean r/NC = {mean_rc:+.4f}, "
                f"median r/NC = {median_rc:+.4f}, "
                f"sig (applied): {n_apply}/{len(nc_subj_pixels)}",
                flush=True,
            )

            nc_log = {
                f"noise_corrected/{subj_name}/mean_r": mean_rc,
                f"noise_corrected/{subj_name}/median_r": median_rc,
                f"noise_corrected/{subj_name}/n_apply_voxels": n_apply,
                f"noise_corrected/{subj_name}/nc_source": nc_subj_source,
                "model_step": step_num,
            }
            if image_shape_ref is not None:
                nc_subj_img = build_brain_map(
                    nc_subj_pixels, image_shape_ref, y_coords_ref, x_coords_ref,
                )
                nc_log[f"noise_corrected/{subj_name}/nc_lower_bound_image"] = fmri_to_wandb_image(
                    nc_subj_img,
                    title=f"{subj_name} within-subject NC lower bound",
                )
                if using_perm:
                    sig_img = build_brain_map(
                        sig_mask_subj.astype(np.float64),
                        image_shape_ref, y_coords_ref, x_coords_ref,
                    )
                    nc_log[f"noise_corrected/{subj_name}/sig_mask_image"] = fmri_to_wandb_image(
                        sig_img,
                        title=f"{subj_name} within-subject sig mask",
                    )

            within_pending.append((nc_log, r_corrected, subj_name))

        # Pass 2: shared colorbar bound across all within-subject r/NC maps,
        # then render + log each with the same `abs_max`.
        within_vmax = nc_colorbar_vmax(args, [rc for _, rc, _ in within_pending])
        for nc_log, r_corrected, subj_name in within_pending:
            if image_shape_ref is not None:
                rc_img = build_brain_map(
                    r_corrected, image_shape_ref, y_coords_ref, x_coords_ref,
                )
                nc_log[f"noise_corrected/{subj_name}/r_image"] = fmri_to_wandb_image(
                    rc_img,
                    title=f"{subj_name} r/within-subject-NC ({model_name})",
                    abs_max=within_vmax,
                )
            wandb.log(nc_log)

        # Per-subject intersubject NC correction — divide each subject's
        # per-voxel r by the *inter-subject* NC lower bound, in addition to
        # the within-subject correction above. The within-subject view answers
        # "how well do we predict this subject given their own reliability
        # ceiling"; the intersubject view normalises by the group-level
        # reliability so subjects are directly comparable on the same axis.
        if nc_intersubj_pixels is not None:
            print(f"\n{'=' * 60}", flush=True)
            print("Per-subject noise-ceiling-corrected r-scores (intersubject NC)", flush=True)
            if intersubj_sig_mask is not None:
                print(
                    f"  sig mask: {intersubj_sig_mask_source} "
                    f"({int(intersubj_sig_mask.sum())}/{len(intersubj_sig_mask)} voxels)",
                    flush=True,
                )
            else:
                print(
                    "  sig mask: none (intersubject permutation file not found "
                    "— applying to every voxel where NC>1e-3)",
                    flush=True,
                )
            print(f"{'=' * 60}", flush=True)
            # Pass 1: compute every subject's intersubject r/NC and stage its
            # scalar payload; defer the r/NC image so all subjects (and the
            # group map below) share one colorbar scale.
            inter_pending = []  # list of (inter_log, r_inter_corrected, subj_name)
            for subj_idx, r_arr in zip(unique_ids, per_subject_r):
                subj_name = (
                    ALL_SUBJECTS[subj_idx]
                    if 0 <= subj_idx < len(ALL_SUBJECTS)
                    else f"subj_id_{subj_idx}"
                )
                if len(nc_intersubj_pixels) != len(r_arr):
                    print(
                        f"  {subj_name}: voxel count mismatch with intersubject NC — skipped.",
                        flush=True,
                    )
                    continue
                if intersubj_sig_mask is not None and len(intersubj_sig_mask) == len(r_arr):
                    effective_mask = intersubj_sig_mask
                else:
                    effective_mask = np.ones_like(r_arr, dtype=bool)
                r_inter_corrected, inter_apply_mask = correct_r_by_nc_with_mask(
                    r_arr, nc_intersubj_pixels, effective_mask,
                )
                per_subject_r_corrected_intersubj[subj_name] = r_inter_corrected
                n_apply = int(inter_apply_mask.sum())
                mean_rc = float(np.nanmean(r_inter_corrected))
                median_rc = float(np.nanmedian(r_inter_corrected))
                print(
                    f"  {subj_name} ({nc_intersubj_source}): mean r/NC = {mean_rc:+.4f}, "
                    f"median r/NC = {median_rc:+.4f}, "
                    f"applied: {n_apply}/{len(nc_intersubj_pixels)}",
                    flush=True,
                )
                inter_log = {
                    f"noise_corrected_intersubj/{subj_name}/mean_r": mean_rc,
                    f"noise_corrected_intersubj/{subj_name}/median_r": median_rc,
                    f"noise_corrected_intersubj/{subj_name}/n_apply_voxels": n_apply,
                    f"noise_corrected_intersubj/{subj_name}/nc_source": nc_intersubj_source,
                    "model_step": step_num,
                }
                inter_pending.append((inter_log, r_inter_corrected, subj_name))

            # Pass 2: shared colorbar bound across all intersubject r/NC maps.
            # The group map (logged later) reuses this same `inter_vmax`.
            if inter_pending:
                inter_vmax = nc_colorbar_vmax(
                    args, [r for _, r, _ in inter_pending]
                )
                for inter_log, r_inter_corrected, subj_name in inter_pending:
                    if image_shape_ref is not None:
                        inter_img = build_brain_map(
                            r_inter_corrected, image_shape_ref, y_coords_ref, x_coords_ref,
                        )
                        inter_log[f"noise_corrected_intersubj/{subj_name}/r_image"] = fmri_to_wandb_image(
                            inter_img,
                            title=f"{subj_name} r/intersubject-NC ({model_name})",
                            abs_max=inter_vmax,
                        )
                    wandb.log(inter_log)

        if nc_intersubj_pixels is None:
            print(
                "\nWARNING: intersubject noise ceiling file not found — group-level "
                "NC correction will be skipped. Expected at "
                f"results/noise_ceiling/noise_ceiling_intersubject_"
                f"{args.data.roi_file}_{args.data.roi}.npz",
                flush=True,
            )

    # Group-level statistics (requires all subjects to share voxel count).
    voxel_counts = {len(r) for r in per_subject_r}
    if len(voxel_counts) != 1:
        print(
            f"\nSkipping group t-test: subjects have heterogeneous voxel counts {voxel_counts}.",
            flush=True,
        )
        return per_subject_mean, None, None

    all_r = np.stack(per_subject_r, axis=0)  # (N_subj, n_voxels)
    n_subjects, n_voxels = all_r.shape

    group_mean_r = np.nanmean(all_r, axis=0)
    overall_mean = float(np.nanmean(group_mean_r))

    t_stats, p_values = ttest_1samp(all_r, popmean=0.0, axis=0, nan_policy="omit")
    # statsmodels-free NaN handling
    p_values = np.asarray(p_values, dtype=np.float64)
    nan_mask = np.isnan(p_values)
    if nan_mask.any():
        p_values[nan_mask] = 1.0
    reject, p_corrected = benjamini_hochberg(p_values, alpha=FDR_ALPHA)
    n_sig = int(reject.sum())

    sig_mean = float(np.nanmean(group_mean_r[reject])) if n_sig > 0 else 0.0
    print(f"\n{'-' * 60}\nGroup statistics across {n_subjects} subjects:", flush=True)
    print(f"  n_voxels                         = {n_voxels}", flush=True)
    print(f"  group mean r (all voxels)        = {overall_mean:+.4f}", flush=True)
    print(f"  group mean r (BH-FDR q<{FDR_ALPHA})    = {sig_mean:+.4f}", flush=True)
    print(
        f"  significant voxels (BH-FDR q<{FDR_ALPHA}) = {n_sig}/{n_voxels} "
        f"({100.0 * n_sig / n_voxels:.1f}%)",
        flush=True,
    )
    print(f"  mean per-subject r               = {np.mean(list(per_subject_mean.values())):+.4f}", flush=True)
    print(f"{'-' * 60}\n", flush=True)

    wandb.log({
        "group/mean_r_all_voxels": overall_mean,
        "group/mean_r_significant": sig_mean,
        "group/n_significant_voxels": n_sig,
        "group/n_total_voxels": int(n_voxels),
        "group/frac_significant": float(n_sig / n_voxels),
        "group/mean_per_subject_r": float(np.mean(list(per_subject_mean.values()))),
        "model_step": step_num,
    })

    # Group-level brain maps (2D only). Two complementary views:
    #   (1) `group/r_image_all_voxels`  — unthresholded cross-subject mean r,
    #       shows the raw effect size landscape across the ROI.
    #   (2) `group/r_image_significant` — same map with non-significant voxels
    #       masked to 0, so only BH-FDR-surviving voxels carry colour.
    # Non-significant voxels map to 0 which renders as neutral on the symmetric
    # RdBu_r colormap (indistinguishable from the ROI background, but that's
    # acceptable here because the thresholded map is paired with the full map).
    if args.data.is_2d and image_shape_ref is not None:
        group_r_img_all = np.zeros(image_shape_ref, dtype=np.float64)
        group_r_img_all[y_coords_ref, x_coords_ref] = np.nan_to_num(group_mean_r, nan=0.0)

        sig_values = np.where(reject, np.nan_to_num(group_mean_r, nan=0.0), 0.0)
        group_r_img_sig = np.zeros(image_shape_ref, dtype=np.float64)
        group_r_img_sig[y_coords_ref, x_coords_ref] = sig_values

        wandb.log({
            "group/r_image_all_voxels": fmri_to_wandb_image(
                group_r_img_all,
                title=f"Group mean r ({n_subjects} subjects, all voxels) — {model_name}",
            ),
            "group/r_image_significant": fmri_to_wandb_image(
                group_r_img_sig,
                title=(
                    f"Group mean r (BH-FDR q<{FDR_ALPHA}, "
                    f"{n_sig}/{n_voxels} sig) — {model_name}"
                ),
            ),
            "model_step": step_num,
        })

    # ── Group-level noise-ceiling correction ──
    # Filter group voxels by BH-FDR (`reject`) only; among the surviving
    # voxels, divide group_mean_r by the *intersubject* NC lower bound per
    # voxel. Voxels failing BH-FDR or with NC<floor stay NaN in the
    # corrected array and are excluded from the reported scalars / images.
    group_r_corrected = None
    if nc_intersubj_pixels is not None and len(nc_intersubj_pixels) == n_voxels:
        group_r_corrected, apply_mask_group = correct_r_by_nc_with_mask(
            group_mean_r, nc_intersubj_pixels, reject,
        )
        n_apply_group = int(apply_mask_group.sum())
        sig_mean_corrected = float(np.nanmean(group_r_corrected)) if n_apply_group > 0 else 0.0

        print(f"\n{'-' * 60}", flush=True)
        print(f"Group noise-ceiling-corrected stats ({nc_intersubj_source}):", flush=True)
        print(
            f"  group mean r/NC (BH-FDR q<{FDR_ALPHA}) = {sig_mean_corrected:+.4f}",
            flush=True,
        )
        print(
            f"  applied to {n_apply_group}/{n_voxels} voxels (BH-FDR rejected)",
            flush=True,
        )
        print(
            f"  mean NC lower bound (all voxels) = "
            f"{float(np.nanmean(nc_intersubj_pixels)):+.4f}",
            flush=True,
        )
        print(f"{'-' * 60}\n", flush=True)

        wandb.log({
            "noise_corrected/group_mean_r_significant": sig_mean_corrected,
            "noise_corrected/n_apply_voxels": n_apply_group,
            "noise_corrected/mean_nc_lower_bound": float(np.nanmean(nc_intersubj_pixels)),
            "noise_corrected/nc_source": nc_intersubj_source,
            "model_step": step_num,
        })

        # Group-level corrected brain map: only BH-rejected & NC-valid voxels
        # carry colour; everything else is rendered as background (0).
        if args.data.is_2d and image_shape_ref is not None:
            group_rc_img_sig = build_brain_map(
                group_r_corrected, image_shape_ref, y_coords_ref, x_coords_ref,
            )
            nc_img = build_brain_map(
                nc_intersubj_pixels, image_shape_ref, y_coords_ref, x_coords_ref,
            )

            # Same colorbar as the per-subject intersubject maps. Reuse the
            # bound set in the per-subject loop; if that loop produced no maps
            # (e.g. all voxel-count mismatched), derive it from the group map.
            group_vmax = inter_vmax if inter_vmax is not None else nc_colorbar_vmax(
                args, [group_r_corrected]
            )
            wandb.log({
                "noise_corrected/r_image_significant": fmri_to_wandb_image(
                    group_rc_img_sig,
                    title=(
                        f"Group r/NC (BH-FDR q<{FDR_ALPHA}, "
                        f"{n_apply_group}/{n_voxels} applied) — {model_name}"
                    ),
                    abs_max=group_vmax,
                ),
                "noise_corrected/nc_lower_bound_image": fmri_to_wandb_image(
                    nc_img,
                    title=f"Intersubject NC lower bound — {args.data.roi_file}_{args.data.roi}",
                ),
                "model_step": step_num,
            })

    # Persist raw per-subject r arrays next to the other generation outputs
    # so downstream analysis mirrors train_sklearn_group.npz format.
    try:
        output_dir = os.path.join(
            getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
            str(args.jobid),
        )
        os.makedirs(output_dir, exist_ok=True)
        safe_model_name = re.sub(r"[^0-9A-Za-z_\-]", "_", str(model_name))
        save_dict = dict(
            all_r=all_r,
            group_mean_r=group_mean_r,
            t_stats=t_stats,
            p_values=p_values,
            p_corrected=p_corrected,
            reject=reject,
            subject_ids=np.array(unique_ids),
            subject_names=np.array(
                [ALL_SUBJECTS[i] if 0 <= i < len(ALL_SUBJECTS) else f"subj_id_{i}" for i in unique_ids]
            ),
        )
        if nc_intersubj_pixels is not None and len(nc_intersubj_pixels) == n_voxels:
            save_dict["nc_lower_bound"] = nc_intersubj_pixels
            save_dict["group_mean_r_corrected"] = group_r_corrected
            save_dict["group_apply_mask"] = apply_mask_group
        # Per-subject corrected r matrices, stacked in the same `unique_ids`
        # order as `all_r` so rows align across all saved arrays.
        subj_name_order = [
            ALL_SUBJECTS[i] if 0 <= i < len(ALL_SUBJECTS) else f"subj_id_{i}"
            for i in unique_ids
        ]

        def _stack_per_subject(corrected_dict):
            rows, kept_names = [], []
            for name in subj_name_order:
                if name in corrected_dict:
                    rows.append(corrected_dict[name])
                    kept_names.append(name)
            if rows and {len(r) for r in rows} == {n_voxels}:
                return np.stack(rows, axis=0), np.array(kept_names)
            return None, None

        intersubj_stack, intersubj_stack_names = _stack_per_subject(
            per_subject_r_corrected_intersubj
        )
        if intersubj_stack is not None:
            save_dict["per_subject_r_corrected_intersubj"] = intersubj_stack
            save_dict["per_subject_r_corrected_intersubj_subjects"] = intersubj_stack_names

        within_stack, within_stack_names = _stack_per_subject(
            per_subject_r_corrected_within
        )
        if within_stack is not None:
            save_dict["per_subject_r_corrected_within"] = within_stack
            save_dict["per_subject_r_corrected_within_subjects"] = within_stack_names
        if intersubj_sig_mask is not None and len(intersubj_sig_mask) == n_voxels:
            save_dict["intersubj_sig_mask"] = intersubj_sig_mask
            save_dict["intersubj_sig_mask_source"] = intersubj_sig_mask_source
        np.savez(
            os.path.join(output_dir, f"group_generation_results_{safe_model_name}.npz"),
            **save_dict,
        )
    except Exception as e:
        print(f"WARNING: failed to save group_generation_results npz: {e}", flush=True)

    return per_subject_mean, overall_mean, n_sig


def _evaluate_single_subject_nc_correction(
    generated_samples_per_model,
    true_fmri,
    args,
    step_num,
    model_name,
):
    """Apply per-subject noise-ceiling correction for single-subject generation.

    Loads ``noise_ceiling_{subj}_{roi_file}_{roi}.npz`` (within-subject
    leave-one-out lower bound, field ``nc_lower_avg_clamped``), projects to
    2D pixel space, computes per-voxel ``r / nc_lower``, and logs corrected
    scalars + brain maps to wandb under the ``noise_corrected/`` prefix.

    Returns (mean_corrected_r, n_voxels_used) or (None, None) on failure.
    """
    if not args.data.is_2d:
        # Single-subject NC correction is only wired for 2D for now (we need
        # the pixel coords to align NC vertices through signal_to_2d).
        print(
            "WARNING: single-subject NC correction skipped — only 2D path "
            "is implemented.",
            flush=True,
        )
        return None, None

    subj = args.data.subj
    y_coords, x_coords = _load_subject_locations_2d(args, subj)

    # Prefer the within-subject permutation NC + significance mask from the
    # aggregated pkl; fall back to the legacy LOO file (no mask filter).
    nc_pixels, sig_mask, nc_source = get_nc_perm_aligned(
        subj, args.data.roi_file, args.data.roi, y_coords, x_coords,
        masks_path=getattr(args.data, "nc_masks_path", NC_MASKS_PATH_DEFAULT),
        mask_key=getattr(args.validation, "nc_mask_key", NC_MASK_KEY_DEFAULT),
    )
    using_perm = nc_pixels is not None
    if not using_perm:
        nc_pixels, nc_source = _load_noise_ceiling_2d_pixels(
            args, y_coords, x_coords, mode="per_subject",
        )
        if nc_pixels is not None:
            sig_mask = np.ones_like(nc_pixels, dtype=bool)
    if nc_pixels is None:
        nc_path = os.path.join(
            "results", "noise_ceiling",
            f"noise_ceiling_{subj}_{args.data.roi_file}_{args.data.roi}.npz",
        )
        print(
            f"\nWARNING: per-subject noise ceiling not found "
            f"(neither {getattr(args.data, 'nc_masks_path', NC_MASKS_PATH_DEFAULT)} "
            f"nor {nc_path}) — skipping single-subject NC correction.",
            flush=True,
        )
        return None, None

    # Convert generated/true to numpy and extract ROI voxel time series.
    gen_np = generated_samples_per_model.cpu().numpy() if isinstance(
        generated_samples_per_model, torch.Tensor) else np.asarray(generated_samples_per_model)
    true_np = true_fmri.cpu().numpy() if isinstance(true_fmri, torch.Tensor) else np.asarray(true_fmri)

    gen_flat = gen_np.squeeze(1) if gen_np.ndim == 4 else gen_np  # (N, H, W)
    true_flat = true_np.squeeze(1) if true_np.ndim == 4 else true_np
    image_shape = gen_flat.shape[1:]

    gen_vox = gen_flat[:, y_coords, x_coords]
    true_vox = true_flat[:, y_coords, x_coords]
    r_per_voxel = per_voxel_r(gen_vox, true_vox)

    r_corrected, apply_mask = correct_r_by_nc_with_mask(
        r_per_voxel, nc_pixels, sig_mask,
    )
    n_apply = int(apply_mask.sum())
    n_total = len(nc_pixels)
    mean_rc = float(np.nanmean(r_corrected))
    median_rc = float(np.nanmedian(r_corrected))
    raw_mean = float(np.nanmean(r_per_voxel))
    nc_mean = float(np.nanmean(nc_pixels))

    print(f"\n{'=' * 60}", flush=True)
    print(f"Single-subject noise-ceiling correction ({subj}) — {model_name}", flush=True)
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
    log_payload = {
        "noise_corrected/mean_r": mean_rc,
        "noise_corrected/median_r": median_rc,
        "noise_corrected/raw_mean_r": raw_mean,
        "noise_corrected/mean_nc_lower_bound": nc_mean,
        "noise_corrected/n_apply_voxels": n_apply,
        "noise_corrected/n_total_voxels": n_total,
        "noise_corrected/frac_apply": float(n_apply / max(n_total, 1)),
        "noise_corrected/nc_source": nc_source,
        "noise_corrected/r_image": fmri_to_wandb_image(
            rc_img, title=f"{subj} r/NC ({model_name})",
            abs_max=nc_colorbar_vmax(args, [r_corrected]),
        ),
        "noise_corrected/nc_lower_bound_image": fmri_to_wandb_image(
            nc_img,
            title=f"NC lower bound ({subj}) — {args.data.roi_file}_{args.data.roi}",
        ),
        "model_step": step_num,
    }
    if using_perm:
        sig_img = build_brain_map(
            sig_mask.astype(np.float64), image_shape, y_coords, x_coords,
        )
        log_payload["noise_corrected/sig_mask_image"] = fmri_to_wandb_image(
            sig_img, title=f"{subj} within-subject sig mask",
        )
    wandb.log(log_payload)

    # ── Intersubject NC correction ──
    # Complements the within-subject correction above by dividing the same
    # per-voxel r by the *inter-subject* NC lower bound (group-level
    # reliability). This puts the single-subject result on the same axis as
    # the multi-subject group correction, so single-subject runs can be
    # compared against multi-subject runs side-by-side.
    nc_intersubj_pixels, nc_intersubj_source = _load_noise_ceiling_2d_pixels(
        args, y_coords, x_coords, mode="intersubject", subj=subj,
    )
    # Intersubject permutation sig mask — only the per-subject correction uses
    # it (the group correction stays gated on BH-FDR; here we're in
    # single-subject context, so there's no group correction at all).
    intersubj_mask_key = getattr(args.validation, "nc_intersubj_mask_key", "sig_mask")
    intersubj_sig_mask, intersubj_sig_mask_source = get_intersubject_sig_mask_aligned(
        args.data.roi_file, args.data.roi, y_coords, x_coords, args,
        mask_key=intersubj_mask_key,
    )
    r_inter_corrected = None
    inter_apply_mask = None
    mean_inter = None
    if nc_intersubj_pixels is None:
        print(
            "  intersubject NC unavailable — skipping intersubject correction "
            f"(expected at results/noise_ceiling/noise_ceiling_intersubject_"
            f"{args.data.roi_file}_{args.data.roi}.npz).",
            flush=True,
        )
    elif len(nc_intersubj_pixels) != len(r_per_voxel):
        print(
            "  intersubject NC voxel-count mismatch — skipping intersubject correction.",
            flush=True,
        )
        nc_intersubj_pixels = None
    else:
        if intersubj_sig_mask is not None and len(intersubj_sig_mask) == len(r_per_voxel):
            effective_mask = intersubj_sig_mask
            mask_label = intersubj_sig_mask_source
        else:
            effective_mask = np.ones_like(r_per_voxel, dtype=bool)
            mask_label = "no sig mask (NC>1e-3 only)"
        r_inter_corrected, inter_apply_mask = correct_r_by_nc_with_mask(
            r_per_voxel, nc_intersubj_pixels, effective_mask,
        )
        n_inter_apply = int(inter_apply_mask.sum())
        mean_inter = float(np.nanmean(r_inter_corrected))
        median_inter = float(np.nanmedian(r_inter_corrected))
        nc_inter_mean = float(np.nanmean(nc_intersubj_pixels))

        print(f"\n{'=' * 60}", flush=True)
        print(f"Single-subject intersubject NC correction ({subj}) — {model_name}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"  NC source              : {nc_intersubj_source}", flush=True)
        print(f"  sig mask               : {mask_label}", flush=True)
        print(
            f"  applied to {n_inter_apply}/{len(nc_intersubj_pixels)} voxels",
            flush=True,
        )
        print(f"  raw mean r              = {raw_mean:+.4f}", flush=True)
        print(f"  mean intersubject NC    = {nc_inter_mean:+.4f}", flush=True)
        print(f"  mean r/NC (corrected)   = {mean_inter:+.4f}", flush=True)
        print(f"  median r/NC (corrected) = {median_inter:+.4f}", flush=True)
        print(f"{'=' * 60}\n", flush=True)

        inter_rc_img = build_brain_map(r_inter_corrected, image_shape, y_coords, x_coords)
        inter_nc_img = build_brain_map(nc_intersubj_pixels, image_shape, y_coords, x_coords)
        wandb.log({
            "noise_corrected_intersubj/mean_r": mean_inter,
            "noise_corrected_intersubj/median_r": median_inter,
            "noise_corrected_intersubj/raw_mean_r": raw_mean,
            "noise_corrected_intersubj/mean_nc_lower_bound": nc_inter_mean,
            "noise_corrected_intersubj/n_apply_voxels": n_inter_apply,
            "noise_corrected_intersubj/n_total_voxels": len(nc_intersubj_pixels),
            "noise_corrected_intersubj/frac_apply": float(
                n_inter_apply / max(len(nc_intersubj_pixels), 1)
            ),
            "noise_corrected_intersubj/nc_source": nc_intersubj_source,
            "noise_corrected_intersubj/r_image": fmri_to_wandb_image(
                inter_rc_img,
                title=f"{subj} r/intersubject-NC ({model_name})",
                abs_max=nc_colorbar_vmax(args, [r_inter_corrected]),
            ),
            "noise_corrected_intersubj/nc_lower_bound_image": fmri_to_wandb_image(
                inter_nc_img,
                title=f"Intersubject NC lower bound — {args.data.roi_file}_{args.data.roi}",
            ),
            "model_step": step_num,
        })

    # Persist alongside other generation artefacts.
    try:
        output_dir = os.path.join(
            getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
            str(args.jobid),
        )
        os.makedirs(output_dir, exist_ok=True)
        safe_model_name = re.sub(r"[^0-9A-Za-z_\-]", "_", str(model_name))
        save_dict = dict(
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
        if r_inter_corrected is not None:
            save_dict["r_corrected_intersubj"] = r_inter_corrected
            save_dict["nc_lower_bound_intersubj"] = nc_intersubj_pixels
            save_dict["apply_mask_intersubj"] = inter_apply_mask
            save_dict["nc_source_intersubj"] = nc_intersubj_source
            if intersubj_sig_mask is not None and len(intersubj_sig_mask) == len(r_per_voxel):
                save_dict["intersubj_sig_mask"] = intersubj_sig_mask
                save_dict["intersubj_sig_mask_source"] = intersubj_sig_mask_source
        np.savez(
            os.path.join(output_dir, f"single_subject_nc_corrected_{safe_model_name}.npz"),
            **save_dict,
        )
    except Exception as e:
        print(f"WARNING: failed to save single_subject_nc_corrected npz: {e}", flush=True)

    return mean_rc, n_apply


def _evaluate_cross_subject_confusion(
    generated_samples_per_model,
    true_fmri,
    subject_ids,
    stim_keys,
    args,
    step_num,
    model_name,
):
    """Compute an N_subj × N_subj confusion matrix of per-voxel Pearson r.

    Entry (i, j) = mean-across-voxels Pearson r between
        predictions generated with identity_label=i, reindexed by stimulus,
    and
        subject j's true fMRI response to the same stimuli.

    Requires stim_keys (act_idx per sample) to align subjects. For the
    averaged variant every subject sees the 515 common_515 stimuli exactly
    once, so each subject's slice has an identical sorted stim_keys set —
    we assert that and index accordingly.

    The diagonal should match per-subject r from
    `_evaluate_per_subject_and_group` (sanity check). The off-diagonal
    measures cross-subject transferability; specificity = mean(diag) - mean(off)
    quantifies whether the identity pathway produces subject-specific outputs.
    """
    gen_np = generated_samples_per_model.cpu().numpy() if isinstance(
        generated_samples_per_model, torch.Tensor) else np.asarray(generated_samples_per_model)
    true_np = true_fmri.cpu().numpy() if isinstance(true_fmri, torch.Tensor) else np.asarray(true_fmri)
    sid_np = subject_ids.cpu().numpy() if isinstance(subject_ids, torch.Tensor) else np.asarray(subject_ids)
    stim_np = stim_keys.cpu().numpy() if isinstance(stim_keys, torch.Tensor) else np.asarray(stim_keys)

    unique_ids = sorted(np.unique(sid_np).tolist())
    n_subj = len(unique_ids)
    subj_names = [
        ALL_SUBJECTS[i] if 0 <= i < len(ALL_SUBJECTS) else f"subj_id_{i}"
        for i in unique_ids
    ]

    # Shared (y_coords, x_coords) for the 2D ROI — identical across all
    # subjects per CLAUDE.md, so pulling from the first subject is safe.
    y_coords = x_coords = None
    if args.data.is_2d:
        y_coords, x_coords = _load_subject_locations_2d(args, subj_names[0])

    # Build (n_subj, n_stim, n_voxels) aligned tensors for predictions and
    # true fMRI. We sort each subject's slice by its act_idx so that the
    # k-th row for every subject corresponds to the same stimulus.
    reference_stim_keys = None
    aligned_pred = []
    aligned_true = []
    for subj_idx in unique_ids:
        mask = sid_np == subj_idx
        stim_s = stim_np[mask]
        order = np.argsort(stim_s, kind="stable")
        stim_sorted = stim_s[order]

        if reference_stim_keys is None:
            reference_stim_keys = stim_sorted
        else:
            if not np.array_equal(stim_sorted, reference_stim_keys):
                # Mismatch means subjects don't share the same stimulus set
                # (e.g. unaveraged variant with per-subject reps). Bail with
                # a clear message instead of silently producing a wrong matrix.
                print(
                    f"[confusion] subject {subj_idx} ({subj_names[unique_ids.index(subj_idx)]}) "
                    f"has different stimulus keys; skipping cross-subject confusion matrix.",
                    flush=True,
                )
                return None

        gen_s = gen_np[mask][order]
        true_s = true_np[mask][order]
        if args.data.is_2d:
            if gen_s.ndim == 4:
                gen_s = gen_s.squeeze(1)
            if true_s.ndim == 4:
                true_s = true_s.squeeze(1)
            gen_vox = gen_s[:, y_coords, x_coords]
            true_vox = true_s[:, y_coords, x_coords]
        else:
            if gen_s.ndim == 3:
                gen_s = gen_s.squeeze(1)
            if true_s.ndim == 3:
                true_s = true_s.squeeze(1)
            gen_vox = gen_s
            true_vox = true_s

        aligned_pred.append(gen_vox)
        aligned_true.append(true_vox)

    aligned_pred = np.stack(aligned_pred, axis=0)  # (n_subj, n_stim, n_vox)
    aligned_true = np.stack(aligned_true, axis=0)
    n_stim = aligned_pred.shape[1]
    n_vox = aligned_pred.shape[2]

    # Fill the confusion matrix.
    confusion = np.full((n_subj, n_subj), np.nan, dtype=np.float64)
    for i_idx in range(n_subj):
        for j_idx in range(n_subj):
            r_per_voxel = per_voxel_r(aligned_pred[i_idx], aligned_true[j_idx])
            confusion[i_idx, j_idx] = float(np.nanmean(r_per_voxel))

    # Print the matrix.
    header = f"Cross-subject confusion matrix — {model_name}"
    print(f"\n{'=' * 60}\n{header}\n{'=' * 60}", flush=True)
    print(
        f"  rows = identity used for generation, cols = true subject",
        flush=True,
    )
    print(
        f"  cell = mean per-voxel Pearson r over {n_stim} shared stimuli, "
        f"{n_vox} voxels",
        flush=True,
    )
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
        f"(larger = identity conditioning produces subject-specific outputs)",
        flush=True,
    )
    print(f"{'=' * 60}\n", flush=True)

    # Log summary scalars + per-cell scalars to wandb under model_step.
    log_payload = {
        "confusion/mean_diagonal": mean_diag,
        "confusion/mean_offdiagonal": mean_off,
        "confusion/specificity": specificity,
        "model_step": step_num,
    }
    for i_idx, i_name in enumerate(subj_names):
        for j_idx, j_name in enumerate(subj_names):
            log_payload[f"confusion/cell/pred_{i_name}_vs_true_{j_name}"] = float(confusion[i_idx, j_idx])

    # Render the matrix as a heatmap figure for wandb (paired with scalars so
    # the time-series is all under one `confusion/` prefix).
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

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
        ax.set_ylabel("generation identity (rows)")
        ax.set_title(
            f"Cross-subject confusion — {model_name}\n"
            f"diag={mean_diag:+.4f}, off={mean_off:+.4f}, spec={specificity:+.4f}"
        )
        for i_idx in range(n_subj):
            for j_idx in range(n_subj):
                val = confusion[i_idx, j_idx]
                if not np.isfinite(val):
                    continue
                # White text when the cell colour is saturated, otherwise black.
                txt_color = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(
                    j_idx, i_idx, f"{val:+.3f}",
                    ha="center", va="center", color=txt_color, fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean per-voxel Pearson r")
        fig.tight_layout()
        log_payload["confusion/matrix_heatmap"] = wandb.Image(fig)
        plt.close(fig)
    except Exception as e:
        print(f"WARNING: failed to render confusion heatmap: {e}", flush=True)

    wandb.log(log_payload)

    # Persist raw matrix + metadata alongside the other generation outputs.
    try:
        output_dir = os.path.join(
            getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
            str(args.jobid),
        )
        os.makedirs(output_dir, exist_ok=True)
        safe_model_name = re.sub(r"[^0-9A-Za-z_\-]", "_", str(model_name))
        np.savez(
            os.path.join(output_dir, f"confusion_matrix_{safe_model_name}.npz"),
            confusion=confusion,
            subject_ids=np.array(unique_ids),
            subject_names=np.array(subj_names),
            stim_keys=reference_stim_keys,
            mean_diagonal=mean_diag,
            mean_offdiagonal=mean_off,
            specificity=specificity,
            n_stim=n_stim,
            n_voxels=n_vox,
        )
    except Exception as e:
        print(f"WARNING: failed to save confusion_matrix npz: {e}", flush=True)

    return confusion


_STEP_TAG_RE = re.compile(r"step_(\d+)")
_MODEL_PREFIXES = ("checkpoint_", "model_checkpoint_", "model_")


def _resolve_model_file(input_folder, suffix):
    """Try 'checkpoint_', 'model_checkpoint_', and 'model_' prefixes; return the first that exists."""
    for prefix in _MODEL_PREFIXES:
        candidate = f"{prefix}{suffix}.pth"
        if os.path.isfile(os.path.join(input_folder, candidate)):
            return candidate
    # Fall back to checkpoint_ prefix (will produce a clear FileNotFoundError later)
    return f"checkpoint_{suffix}.pth"


def _infer_wandb_step(model_name, args, model_dir=None, checkpoint_step=None):
    """Return an integer step for wandb logging.

    Prefers the real training step stored inside the checkpoint (read at load
    time and passed in as ``checkpoint_step``). Falls back to parsing the
    filename (``step_1234``), then to scanning the directory for the latest
    numbered checkpoint when the tag is ``final``. Only returns None if no
    source is available.
    """
    if checkpoint_step is not None:
        try:
            return int(checkpoint_step)
        except (TypeError, ValueError):
            pass

    match = _STEP_TAG_RE.search(str(model_name))
    if match:
        return int(match.group(1))

    if str(model_name).lower() == "final" and model_dir and os.path.isdir(model_dir):
        step_nums = []
        for fname in os.listdir(model_dir):
            m = _STEP_TAG_RE.search(fname)
            if m:
                step_nums.append(int(m.group(1)))
        if step_nums:
            return args.validation.final_model_num #max(step_nums) + 1  # final model is one step after the last checkpoint

    return None


def _model_file_sort_key(filename):
    name = str(filename)
    if "final" in name.lower():
        return (2, float("inf"), name)
    if "best" in name.lower():
        return (3, float("inf"), name)
    match = _STEP_TAG_RE.search(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, float("inf"), name)

def _infer_model_dims_from_dataloader(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        fmri_data = getattr(dataset, "fmri_data", None)
        activations = getattr(dataset, "activations", None)
        if fmri_data is not None and activations is not None:
            return fmri_data.shape[1:], activations.shape[-1] # to cater for 2D cases

    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        fmri_signal, cond_signal = batch[0], batch[1]
        return fmri_signal.shape[1:], cond_signal.shape[-1]

    raise ValueError("Expected dataloader to yield (fmri_signal, cond) pairs.")

def generate_sample_loop_toy(args):
    print("Setting up diffusion process...", flush=True)
    diffusion_process = get_diffusion(args, device=DEVICE)
    generated_samples = {}

    if str(args.model.which).lower() == "all":
        print("Testing all the models in the model folder...")
        model_files = [f for f in os.listdir(args.model.input_folder) if f.endswith('.pth')]
        model_files = sorted(model_files, key=_model_file_sort_key)
    else:
        print(f"Testing the model at step {args.model.which}")
        if str(args.model.which).lower() == "final":
            model_files = [_resolve_model_file(args.model.input_folder, "final")]
        elif str(args.model.which).lower() == "best":
            model_files = [_resolve_model_file(args.model.input_folder, "best")]
        else:
            model_files = [_resolve_model_file(args.model.input_folder, f"step_{args.model.which}")]

    for model_file in model_files:
        print("Setting up the model: ", model_file)
        model_path = os.path.join(args.model.input_folder, model_file)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = set_model(args)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Sometimes the checkpoint IS the state_dict itself
            model.load_state_dict(checkpoint)
        print(f"Starting generation. We generate label {args.validation.label_to_generate}")
        generated_samples_one_model = generate_samples(args.validation.batch_size, model, diffusion_process, args, device=DEVICE) * 255
        generated_samples_one_model = torch.clip(generated_samples_one_model, 0, 255)    
        print("max value of generated samples: ", generated_samples_one_model.max().item())
        print("min value of generated samples: ", generated_samples_one_model.min().item())

        filename = os.path.basename(model_path)
        match = re.search(r"(step_\d+|final|best)", filename)
        tag = match.group(1) if match else filename
        generated_samples[f"{tag}"] = generated_samples_one_model

    return generated_samples

def _log_per_subject_trajectory_gif(
    args,
    model,
    diffusion_process,
    ann_tokenizer,
    cond_concat,
    true_fmri_concat,
    subject_ids_concat,
    model_name,
    device,
    stim_keys_concat=None,
):
    """For each unique subject, run one focused denoising pass with trajectory
    tracking on a single stimulus belonging to that subject, then render a GIF
    where every frame shows the generated sample at SDE time t alongside the
    fixed ground-truth fMRI for that stimulus.
    """
    if not getattr(args.data, "is_2d", False):
        return
    if subject_ids_concat is None or cond_concat is None:
        return
    try:
        from io import BytesIO
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from PIL import Image

        sid_np = (
            subject_ids_concat.cpu().numpy()
            if isinstance(subject_ids_concat, torch.Tensor)
            else np.asarray(subject_ids_concat)
        )
        # First-occurrence index per subject in the concatenated test stream.
        seen = set()
        sel_indices = []
        for i, s in enumerate(sid_np.tolist()):
            if s not in seen:
                seen.add(s)
                sel_indices.append(i)
        if not sel_indices:
            return

        sel_cond = cond_concat[sel_indices].to(device)
        sel_true = true_fmri_concat[sel_indices]
        sel_sid_t = subject_ids_concat[sel_indices].to(device)
        if stim_keys_concat is not None:
            sel_stim = (
                stim_keys_concat.cpu().numpy()
                if isinstance(stim_keys_concat, torch.Tensor)
                else np.asarray(stim_keys_concat)
            )[sel_indices]
        else:
            sel_stim = None
        cross_subject_eval = bool(getattr(args.data, "cross_subject_eval", False))
        identity_label = None if cross_subject_eval else sel_sid_t

        # Single focused generation pass with full-batch trajectory tracking.
        _, trajectory = generate_samples(
            sel_cond.shape[0], model, diffusion_process, args,
            cond=sel_cond, device=device,
            ann_tokenizer=ann_tokenizer, identity_label=identity_label,
            return_trajectory=True,
        )
        if not trajectory:
            print("[per-subject-traj-gif] empty trajectory returned, skipping", flush=True)
            return

        true_np = sel_true.cpu().numpy() if isinstance(sel_true, torch.Tensor) else np.asarray(sel_true)
        if true_np.ndim == 4:
            true_np = true_np.squeeze(1)

        safe_model_name = re.sub(r"[^0-9A-Za-z_\-]", "_", str(model_name))
        gif_dir = os.path.join(
            getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
            str(getattr(args, "jobid", "per_subject_traj_gif_debug")),
        )
        os.makedirs(gif_dir, exist_ok=True)

        sel_sid_np = sel_sid_t.cpu().numpy().tolist()
        for slot, subj_idx_int in enumerate(sel_sid_np):
            subj_name = (
                ALL_SUBJECTS[subj_idx_int]
                if 0 <= subj_idx_int < len(ALL_SUBJECTS)
                else f"subj_id_{subj_idx_int}"
            )
            gt_img = true_np[slot]
            true_vmax = max(abs(float(gt_img.min())), abs(float(gt_img.max())), 1e-8)

            frames = []
            for t_val, snap in trajectory:
                gen_img = snap[slot]
                # snap may be (B, 1, H, W) or (B, H, W); collapse channel
                if gen_img.ndim == 3:
                    gen_img = gen_img[0]
                gen_vmax = max(abs(float(gen_img.min())), abs(float(gen_img.max())), 1e-8)

                fframe, (ax_g, ax_t) = plt.subplots(1, 2, figsize=(8, 4))
                ax_g.imshow(
                    gen_img, cmap="RdBu_r", origin="lower",
                    norm=TwoSlopeNorm(vmin=-gen_vmax, vcenter=0.0, vmax=gen_vmax),
                )
                ax_g.set_title(f"{subj_name} gen, t={t_val:.3f}")
                ax_g.set_xticks([]); ax_g.set_yticks([])
                ax_t.imshow(
                    gt_img, cmap="RdBu_r", origin="lower",
                    norm=TwoSlopeNorm(vmin=-true_vmax, vcenter=0.0, vmax=true_vmax),
                )
                ax_t.set_title(f"{subj_name} ground truth")
                ax_t.set_xticks([]); ax_t.set_yticks([])
                fframe.tight_layout()

                buf = BytesIO()
                fframe.savefig(buf, format="png", dpi=80, bbox_inches="tight")
                plt.close(fframe)
                buf.seek(0)
                frames.append(Image.open(buf).convert("RGB"))

            gif_path = os.path.join(
                gif_dir,
                f"per_subject_trajectory_{subj_name}_{safe_model_name}.gif",
            )
            durations = [300] * (len(frames) - 1) + [10000]
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=1,
                optimize=False,
            )
            wandb.log({
                f"per_subject/{subj_name}/trajectory_gif": wandb.Video(
                    gif_path, fps=4, format="gif"
                ),
            })
            print(
                f"[per-subject-traj-gif] {subj_name}: {len(frames)}-frame trajectory GIF saved to {gif_path}",
                flush=True,
            )

        # Dump everything a post-hoc baseline (e.g. Ridge) needs to re-render
        # the same per-subject trajectory side-by-side with its own prediction.
        # Captures the exact stimuli/ANN/identity/trajectory used here.
        try:
            traj_t = np.array([float(tv) for tv, _ in trajectory], dtype=np.float32)
            traj_snaps = np.stack(
                [np.asarray(snap) for _, snap in trajectory], axis=0
            )  # (T, B, ...)
            sel_ann = sel_cond.detach().cpu().numpy()
            sel_true_np_save = sel_true.cpu().numpy() if isinstance(sel_true, torch.Tensor) else np.asarray(sel_true)
            subj_names_arr = np.array(
                [
                    ALL_SUBJECTS[s] if 0 <= s < len(ALL_SUBJECTS) else f"subj_id_{s}"
                    for s in sel_sid_np
                ]
            )
            inputs_path = os.path.join(
                gif_dir, f"trajectory_inputs_{safe_model_name}.npz"
            )
            save_kwargs = dict(
                trajectory_t=traj_t,
                trajectory_snapshots=traj_snaps,
                selected_subject_ids=np.asarray(sel_sid_np, dtype=np.int64),
                selected_subject_names=subj_names_arr,
                selected_ann=sel_ann,
                selected_true_fmri=sel_true_np_save,
                roi=np.array(args.data.roi),
                roi_file=np.array(args.data.roi_file),
                grid_resolution_2d=np.array(args.data.grid_resolution_2d),
                model_name=np.array(str(model_name)),
            )
            if sel_stim is not None:
                save_kwargs["selected_stim_keys"] = np.asarray(sel_stim)
                # Resolve NSD image IDs for each selected stim_key, using the
                # *exact* ordering the activations cache was built from — the
                # two loaders disagree:
                #   - single-subject `get_ann_brain_dataloader`: cache is built
                #     from `test_nsd_ids` returned by `get_train_test_indices`
                #     (common_515 file order in averaged mode).
                #   - multi-subject `get_ann_brain_dataloader_multisubject`:
                #     cache is built from `union_test_nsd_ids = np.sort(...)`
                #     (sorted union). `act_idx` indexes into THIS sorted array.
                # Using the wrong source would produce silently misaligned
                # NSD ids and wrong Ridge lookups downstream.
                try:
                    is_multi_subject = bool(getattr(args.data, "multi_subject", False))
                    if is_multi_subject:
                        from diffusion_brain.utils.fmri_behav_data_utils import get_multi_subject_train_test_indices
                        ms_info = get_multi_subject_train_test_indices(args)
                        nsd_id_source = np.asarray(ms_info["union_test_nsd_ids"])
                        nsd_source_label = "union_test_nsd_ids (sorted)"
                    else:
                        from diffusion_brain.utils.fmri_behav_data_utils import get_train_test_indices
                        _, test_nsd_ids_full, _, _ = get_train_test_indices(args)
                        nsd_id_source = np.asarray(test_nsd_ids_full)
                        nsd_source_label = "test_nsd_ids (single-subject, common_515 order)"
                    save_kwargs["selected_nsd_ids"] = nsd_id_source[np.asarray(sel_stim, dtype=np.int64)]
                    save_kwargs["test_nsd_ids"] = nsd_id_source
                    save_kwargs["test_nsd_ids_source"] = np.array(nsd_source_label)
                except Exception as nsd_e:
                    print(
                        f"[per-subject-traj-gif] could not resolve NSD ids for stim_keys "
                        f"(comparator will reject this file): {nsd_e}",
                        flush=True,
                    )
            np.savez(inputs_path, **save_kwargs)
            print(
                f"[per-subject-traj-gif] inputs npz saved to {inputs_path} "
                f"(stim_keys: {sel_stim is not None}, "
                f"nsd_ids: {'selected_nsd_ids' in save_kwargs})",
                flush=True,
            )
        except Exception as save_e:
            print(f"[per-subject-traj-gif] failed to save inputs npz: {save_e}", flush=True)
    except Exception as e:
        print(f"[per-subject-traj-gif] Failed: {e}", flush=True)


def generate_sample_loop(args):
    print("Setting up true samples and conditions dataloader for comparison...", flush=True)
    print("In generation we use the data from subject: ", args.data.subj)
    _, gen_dataloader = get_dataloader(args)
    input_size, cross_attention_dim = _infer_model_dims_from_dataloader(gen_dataloader)
    args.model.input_size = tuple(input_size) # was int(input_size)

    # fmri_min/fmri_max for thresholding: prefer checkpoint (train set), fallback to current dataset
    gen_ds = gen_dataloader.dataset
    args.data.fmri_min = getattr(gen_ds, "fmri_min", None)
    args.data.fmri_max = getattr(gen_ds, "fmri_max", None)
    
    # Apply the same conditioning tokenization as in training
    ann_dim = int(cross_attention_dim)
    args.model.ann_dim = ann_dim
    cond_token_mode = getattr(args.model, "cond_token_mode", "learned")

    if args.model.condition_mode == "additive":
        print(f"Using additive conditioning, ANN dim {ann_dim} will be added to time embedding")
        args.model.cross_attention_dim = ann_dim
    else:
        num_tokens = max(1, int(getattr(args.model, "cond_seq_len", 8)))
        token_dim = int(getattr(args.model, "token_dim", 256))

        if cond_token_mode == "learned":
            args.model.cross_attention_dim = token_dim
            print(
                f"Condition tokenization: LEARNED | ANN dim {ann_dim} -> "
                f"ANNTokenizer({num_tokens} tokens x {token_dim}-dim), "
                f"cross_attention_dim={token_dim}",
                flush=True,
            )
        else:
            cond_seq_len = num_tokens
            if cond_token_mode == "chunk" and cond_seq_len > 1 and ann_dim % cond_seq_len == 0:
                args.model.cross_attention_dim = ann_dim // cond_seq_len
                print(f"Condition tokenization: chunk | ANN dim {ann_dim} -> seq_len {cond_seq_len} x token_dim {args.model.cross_attention_dim}", flush=True)
            else:
                args.model.cross_attention_dim = ann_dim
                print(f"Condition tokenization: repeat | seq_len {cond_seq_len}, token_dim {args.model.cross_attention_dim}", flush=True)

    args.model.input_folder = args.model.input_folder + "-" + str(args.model.run_id) 

    print("Setting up diffusion process...", flush=True)
    diffusion_process = get_diffusion(args, device=DEVICE)
    generated_samples = {}
    true_fmri_per_model = {}
    model_step_per_model = {}  # tag -> training step read from checkpoint

    if str(args.model.which).lower() == "all":
        print("Testing all the models in the model folder...")
        model_files = [f for f in os.listdir(args.model.input_folder) if f.endswith('.pth')]
        model_files = sorted(model_files, key=_model_file_sort_key)
    elif isinstance(args.model.which, (list, tuple)):
        print(f"Testing the models at steps {args.model.which}")
        model_files = []
        for step in args.model.which:
            if str(step).lower() == "final":
                model_files.append(_resolve_model_file(args.model.input_folder, "final"))
            elif str(step).lower() == "best":
                model_files.append(_resolve_model_file(args.model.input_folder, "best"))
            else:
                model_files.append(_resolve_model_file(args.model.input_folder, f"step_{step}"))
    else:
        print(f"Testing the model at step {args.model.which}")
        if str(args.model.which).lower() == "final":
            model_files = [_resolve_model_file(args.model.input_folder, "final")]
        elif str(args.model.which).lower() == "best":
            model_files = [_resolve_model_file(args.model.input_folder, "best")]
        else:
            model_files = [_resolve_model_file(args.model.input_folder, f"step_{args.model.which}")]

    # Create ANNTokenizer if using learned conditioning
    ann_tokenizer = None
    _1d_cross_attn = args.model.name == "gfdm-unet-1d-cond" and getattr(args.model, "condition_mode", "additive") == "cross_attention"
    _condition_mode = getattr(args.model, "condition_mode", "cross_attention")
    _needs_tokenizer = (args.model.name == "unet-diffusers" and _condition_mode != "additive") or _1d_cross_attn
    if cond_token_mode == "learned" and _needs_tokenizer:
        ann_tokenizer = ANNTokenizer(ann_dim=ann_dim, num_tokens=num_tokens, token_dim=token_dim).to(DEVICE)
        print(f"ANNTokenizer created for generation: {ann_dim} -> {num_tokens} tokens x {token_dim}-dim")
    elif args.model.name == "unet-diffusers" and _condition_mode == "additive":
        print(f"Additive conditioning mode: ANNTokenizer not needed for generation")

    for model_file in model_files:
        print("Setting up the model: ", model_file)
        model_path = os.path.join(args.model.input_folder, model_file)
        print("model path is ", model_path)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # Read the true training step from the checkpoint itself — this is
        # authoritative for "best"/"final" tags where the filename doesn't
        # encode a step number. Falls back to None for legacy checkpoints
        # that don't carry a "step" field.
        step_from_checkpoint = checkpoint.get("step") if isinstance(checkpoint, dict) else None
        model = set_model(args)

        # Support both bundled checkpoint format (checkpoint_*.pth) and
        # legacy format (model_*.pth with separate ann_tokenizer_*.pth)
        if 'model_state_dict' in checkpoint:
            # New bundled checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])
            if ann_tokenizer is not None and 'ann_tokenizer_state_dict' in checkpoint:
                ann_tokenizer.load_state_dict(checkpoint['ann_tokenizer_state_dict'])
                print(f"Loaded ANNTokenizer from bundled checkpoint {model_file}")
                ann_tokenizer.eval()

            # If EMA weights are available, load them into the model for generation
            # (EMA weights are smoother and produce better samples)
            if 'ema_state_dict' in checkpoint:
                all_params = list(model.parameters())
                if ann_tokenizer is not None:
                    all_params += list(ann_tokenizer.parameters())
                ema = EMAModel(all_params, decay=checkpoint['ema_state_dict']['decay'])
                ema.load_state_dict(checkpoint['ema_state_dict'])
                ema.copy_to(all_params)
                print(f"Loaded EMA weights (decay={ema.decay}, updates={ema.num_updates}) for generation")
        else:
            # Legacy format: model weights only
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Load matching ANNTokenizer checkpoint (legacy separate file)
            if ann_tokenizer is not None:
                tokenizer_file = model_file.replace("model_", "ann_tokenizer_")
                tokenizer_path = os.path.join(args.model.input_folder, tokenizer_file)
                if os.path.isfile(tokenizer_path):
                    ann_tokenizer.load_state_dict(torch.load(tokenizer_path, map_location=DEVICE))
                    print(f"Loaded ANNTokenizer from {tokenizer_path}")
                else:
                    print(f"WARNING: ANNTokenizer checkpoint not found at {tokenizer_path}")
                ann_tokenizer.eval()

        # Load normalised fMRI range from checkpoint (train set) for thresholding
        if "fmri_min" in checkpoint and "fmri_max" in checkpoint:
            args.data.fmri_min = checkpoint["fmri_min"]
            args.data.fmri_max = checkpoint["fmri_max"]
            print(f"Loaded fMRI thresholding range from checkpoint: [{args.data.fmri_min:.4f}, {args.data.fmri_max:.4f}]", flush=True)

        model.eval()  # set to eval mode for generation
        print("model device: ", next(model.parameters()).device)
        print("len dataloader.dataset is ", len(gen_dataloader.dataset))


        generated_samples_list = []
        true_fmri_list = []
        subject_ids_list = []
        stim_keys_list = []  # act_idx per sample; enables cross-subject alignment
        cond_list = []  # ANN conditioning per sample; used for the per-subject trajectory GIF

        # Cross-subject evaluation: load multi-subject test data for per-subject
        # metrics, but don't pass identity_label to the model (it wasn't trained
        # with subject conditioning). Set data.cross_subject_eval: true in config.
        cross_subject_eval = bool(getattr(args.data, "cross_subject_eval", False))
        if cross_subject_eval:
            print("Cross-subject evaluation mode: identity_label will NOT be passed to the model", flush=True)

        for idx, batch in enumerate(gen_dataloader):
            # Dataset returns 2-tuple (fmri, ann) for single-subject, or
            # 4-tuple (fmri, ann, subject_id, act_idx) for multi-subject.
            true_fmri, cond = batch[0], batch[1]
            subject_id = batch[2] if len(batch) > 2 else None
            stim_key = batch[3] if len(batch) > 3 else None
            # Pass identity to model only if trained with it (not cross-subject eval)
            identity_label = None if cross_subject_eval else (subject_id.to(DEVICE) if subject_id is not None else None)
            cond = cond.to(DEVICE)

            print(f"Generating samples with guidance strength {args.validation.guidance_scale} for batch {idx+1} out of {len(gen_dataloader)}...", flush=True)
            print(f"In this generation procedure we will average across {args.validation.average_over_num_runs} runs")

            generated_samples_one_model_one_cond = []
            for _ in range(args.validation.average_over_num_runs):
                generated_samples_one_model_one_cond_one_time = generate_samples(args.validation.batch_size, model, diffusion_process, args, cond=cond, device=DEVICE, ann_tokenizer=ann_tokenizer, identity_label=identity_label)
                generated_samples_one_model_one_cond.append(generated_samples_one_model_one_cond_one_time)
            generated_samples_one_model_one_cond = torch.stack(generated_samples_one_model_one_cond, dim=0).mean(dim=0) # averaging across repetitions of the same generation, conditioned on the same ANN signal

            print("Shape of generated samples after stacking average_over_num_runs: ", generated_samples_one_model_one_cond.shape)

            autoencoder = get_linear_autoencoder(args)
            if autoencoder is not None:
                autoencoder.to(DEVICE)
                with torch.no_grad():
                    z = generated_samples_one_model_one_cond.squeeze(1)
                    generated_samples_one_model_one_cond = autoencoder.decoder(z)

            print("max value of generated samples: ", generated_samples_one_model_one_cond.max().item())
            print("min value of generated samples: ", generated_samples_one_model_one_cond.min().item())
            
            generated_samples_one_model_one_cond = generated_samples_one_model_one_cond.squeeze(1)
            print("Dimension of generaetd samples: ", generated_samples_one_model_one_cond.shape, flush=True)

            # Log a few sample images to wandb per batch for visual monitoring
            # if idx == 0:
            #     n_preview = min(4, generated_samples_one_model_one_cond.shape[0])
            #     preview_images = {}
            #     for i in range(n_preview):
            #         gen_np = generated_samples_one_model_one_cond[i].cpu().numpy()
            #         true_np = true_fmri[i].squeeze().cpu().numpy()
            #         preview_images[f"preview/generated_{i}"] = fmri_to_wandb_image(gen_np, title=f"Generated {i}")
            #         preview_images[f"preview/true_{i}"] = fmri_to_wandb_image(true_np, title=f"True {i}")
            #     wandb.log(preview_images)

            generated_samples_list.append(generated_samples_one_model_one_cond.cpu())
            true_fmri_list.append(true_fmri.cpu())
            cond_list.append(cond.cpu())
            if subject_id is not None:
                subject_ids_list.append(subject_id.cpu())
            if stim_key is not None:
                stim_keys_list.append(stim_key.cpu())

        # Concatenate all batches
        generated_samples_one_model = torch.cat(generated_samples_list, dim=0)
        true_fmri_concat = torch.cat(true_fmri_list, dim=0)
        subject_ids_concat = torch.cat(subject_ids_list, dim=0) if subject_ids_list else None
        stim_keys_concat = torch.cat(stim_keys_list, dim=0) if stim_keys_list else None
        cond_concat = torch.cat(cond_list, dim=0) if cond_list else None

        print("Shape of generated samples after concatenating batches: ", generated_samples_one_model.shape, flush=True)
        print("Shape of true fMRI after concatenating batches: ", true_fmri_concat.shape, flush=True)

        filename = os.path.basename(model_path)
        match = re.search(r"(step_\d+|final|best)", filename)
        tag = match.group(1) if match else filename
        generated_samples[f"{tag}"] = generated_samples_one_model
        true_fmri_per_model[f"{tag}"] = (true_fmri_concat, subject_ids_concat, stim_keys_concat)
        model_step_per_model[f"{tag}"] = step_from_checkpoint

        # Per-subject trajectory GIFs (one focused generation pass per model)
        if subject_ids_concat is not None and cond_concat is not None:
            _log_per_subject_trajectory_gif(
                args=args,
                model=model,
                diffusion_process=diffusion_process,
                ann_tokenizer=ann_tokenizer,
                cond_concat=cond_concat,
                true_fmri_concat=true_fmri_concat,
                subject_ids_concat=subject_ids_concat,
                model_name=tag,
                device=DEVICE,
                stim_keys_concat=stim_keys_concat,
            )

    return generated_samples, true_fmri_per_model, model_step_per_model

def main(): 
    args = parse_args_and_setup_wandb()
    print("ARGS: ", args)
    
    ### seed the generation ####
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #####

    wandb.define_metric("model_step")
    wandb.define_metric("*", step_metric="model_step")
    if args.data.data_name == "ann-brain":
        print("Using ANN-Brain dataset, we will generate samples and visualise them on the brain surface.")
        generated_samples, true_fmri_per_model, model_step_per_model = generate_sample_loop(args)
        print(f"Generating the visualisations with guidance scale {args.validation.guidance_scale})")
        for model_name, generated_samples_per_model in generated_samples.items():
            print("Visualising results for model at step: ", model_name)
            step_num = _infer_wandb_step(
                model_name,
                args,
                model_dir=args.model.input_folder,
                checkpoint_step=model_step_per_model.get(model_name),
            )
            print(f"  → wandb model_step = {step_num} (from {'checkpoint' if model_step_per_model.get(model_name) is not None else 'filename/fallback'})", flush=True)
            true_fmri, subject_ids, stim_keys = true_fmri_per_model[model_name]
            visualise_and_save_results(
                generated_samples_per_model,
                true_fmri=true_fmri,
                step=model_name,
                args=args,
                step_num=step_num,
            )

            # Per-subject evaluation for multi-subject models. Mirrors the
            # output format of train_sklearn_group.py (per-subject Pearson r,
            # group mean r, BH-FDR corrected significance) so generation
            # performance can be compared directly to the Ridge baseline.
            if subject_ids is not None:
                _evaluate_per_subject_and_group(
                    generated_samples_per_model,
                    true_fmri,
                    subject_ids,
                    args,
                    step_num,
                    model_name,
                )
            else:
                # Single-subject path: apply per-subject noise-ceiling
                # correction using noise_ceiling_{subj}_{roi_file}_{roi}.npz.
                _evaluate_single_subject_nc_correction(
                    generated_samples_per_model,
                    true_fmri,
                    args,
                    step_num,
                    model_name,
                )

            # Cross-subject confusion matrix: (i, j) = mean per-voxel Pearson r
            # between predictions generated with identity=i and subject j's
            # true fMRI. Requires stim_keys to align subjects by stimulus.
            # Gated by validation.cross_subject_confusion (default true when
            # stim_keys are present).
            run_confusion = bool(getattr(args.validation, "cross_subject_confusion", True))
            if run_confusion and subject_ids is not None and stim_keys is not None:
                _evaluate_cross_subject_confusion(
                    generated_samples_per_model,
                    true_fmri,
                    subject_ids,
                    stim_keys,
                    args,
                    step_num,
                    model_name,
                )
            
            if not args.data.is_2d:
                # I want to see non-interpolated on pycortex flatmap images!
                one_generated_sample_2d, _ = signal_to_2d(args, one_signal_to_transform=generated_samples_per_model[0])
                one_fmri_signal_2d, _ = signal_to_2d(args, one_signal_to_transform=true_fmri[0])

                wandb.log({
                    "true_fmri_data": fmri_to_wandb_image(one_fmri_signal_2d, title="True fMRI"),
                    "generated_data": fmri_to_wandb_image(one_generated_sample_2d, title="Generated"),
                })

            # Save raw generated samples + true fMRI for downstream analysis
            try:
                output_dir = os.path.join(
                    getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
                    str(args.jobid),
                )
                os.makedirs(output_dir, exist_ok=True)
                safe_model_name = re.sub(r"[^0-9A-Za-z_\-]", "_", str(model_name))
                save_dict = {
                    "generated": generated_samples_per_model.cpu().numpy()
                        if isinstance(generated_samples_per_model, torch.Tensor)
                        else np.asarray(generated_samples_per_model),
                    "true_fmri": true_fmri.cpu().numpy()
                        if isinstance(true_fmri, torch.Tensor)
                        else np.asarray(true_fmri),
                }
                if subject_ids is not None:
                    save_dict["subject_ids"] = subject_ids.cpu().numpy() if isinstance(subject_ids, torch.Tensor) else np.asarray(subject_ids)
                if stim_keys is not None:
                    save_dict["stim_keys"] = stim_keys.cpu().numpy() if isinstance(stim_keys, torch.Tensor) else np.asarray(stim_keys)
                if step_num is not None:
                    save_dict["model_step"] = np.array(step_num)
                np.savez(
                    os.path.join(output_dir, f"generated_samples_{safe_model_name}.npz"),
                    **save_dict,
                )
                print(f"Saved generated samples to {output_dir}/generated_samples_{safe_model_name}.npz", flush=True)
            except Exception as e:
                print(f"WARNING: failed to save generated samples: {e}", flush=True)

            # no f-string in the name as i want to have all the models in the slide bar in wandb
            #pyplot_brain(generated_samples_per_model.mean(axis=0), args=args, savename=f"generated_samples_mean", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)
    else:
        print(f"Using {args.data.data_name} dataset, we will generate samples and visualise them as images.")
        generated_samples = generate_sample_loop_toy(args)
    
        print(f"Generating the visualisations with guidance scale {args.validation.guidance_scale})")
        for model_name, generated_samples_per_model in generated_samples.items():
            print("Visualising results for model at step: ", model_name)
            step_num = _infer_wandb_step(model_name, args, model_dir=args.model.input_folder)
            visualise_and_save_results(generated_samples_per_model, step=model_name, args=args, step_num=step_num)

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cpu"
    main()   
