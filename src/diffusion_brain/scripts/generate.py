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

FDR_ALPHA = 0.05


def _benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (reject, p_corrected).

    Mirrors the implementation in train_sklearn_group.py so per-subject
    generation stats are directly comparable to the Ridge baseline.
    """
    m = len(p_values)
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]

    p_corrected = np.empty(m)
    cummin = 1.0
    for i in range(m - 1, -1, -1):
        adjusted = sorted_p[i] * m / (i + 1)
        cummin = min(cummin, adjusted)
        p_corrected[sort_idx[i]] = min(cummin, 1.0)

    reject = p_corrected <= alpha
    return reject, p_corrected


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


def _per_voxel_r(pred, true):
    """Per-voxel Pearson r. Inputs: (N_images, n_voxels) numpy arrays."""
    n_voxels = pred.shape[1]
    r = np.full(n_voxels, np.nan, dtype=np.float64)
    for v in range(n_voxels):
        p = pred[:, v]
        t = true[:, v]
        # pearsonr raises on zero variance — guard to keep going
        if np.std(p) == 0 or np.std(t) == 0:
            continue
        r[v] = pearsonr(t, p)[0]
    return r


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

        r_per_voxel = _per_voxel_r(gen_vox, true_vox)

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
    reject, p_corrected = _benjamini_hochberg(p_values, alpha=FDR_ALPHA)
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

    # Persist raw per-subject r arrays next to the other generation outputs
    # so downstream analysis mirrors train_sklearn_group.npz format.
    try:
        output_dir = os.path.join(
            getattr(args.validation, "output_folder", "./model_outputs/ann-brain/"),
            str(args.jobid),
        )
        os.makedirs(output_dir, exist_ok=True)
        safe_model_name = re.sub(r"[^0-9A-Za-z_\-]", "_", str(model_name))
        np.savez(
            os.path.join(output_dir, f"group_generation_results_{safe_model_name}.npz"),
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
    except Exception as e:
        print(f"WARNING: failed to save group_generation_results npz: {e}", flush=True)

    return per_subject_mean, overall_mean, n_sig


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

        for idx, batch in enumerate(gen_dataloader):
            # Dataset returns 2-tuple (fmri, ann) or 3-tuple (fmri, ann, subject_id)
            true_fmri, cond = batch[0], batch[1]
            identity_label = batch[2].to(DEVICE) if len(batch) > 2 else None
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
            if identity_label is not None:
                subject_ids_list.append(identity_label.cpu())

        # Concatenate all batches
        generated_samples_one_model = torch.cat(generated_samples_list, dim=0)
        true_fmri_concat = torch.cat(true_fmri_list, dim=0)
        subject_ids_concat = torch.cat(subject_ids_list, dim=0) if subject_ids_list else None

        print("Shape of generated samples after concatenating batches: ", generated_samples_one_model.shape, flush=True)
        print("Shape of true fMRI after concatenating batches: ", true_fmri_concat.shape, flush=True)

        filename = os.path.basename(model_path)
        match = re.search(r"(step_\d+|final|best)", filename)
        tag = match.group(1) if match else filename
        generated_samples[f"{tag}"] = generated_samples_one_model
        true_fmri_per_model[f"{tag}"] = (true_fmri_concat, subject_ids_concat)
        model_step_per_model[f"{tag}"] = step_from_checkpoint

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
            true_fmri, subject_ids = true_fmri_per_model[model_name]
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
            
            if not args.data.is_2d:
                # I want to see non-interpolated on pycortex flatmap images!
                one_generated_sample_2d, _ = signal_to_2d(args, one_signal_to_transform=generated_samples_per_model[0])
                one_fmri_signal_2d, _ = signal_to_2d(args, one_signal_to_transform=true_fmri[0])

                wandb.log({
                    "true_fmri_data": fmri_to_wandb_image(one_fmri_signal_2d, title="True fMRI"),
                    "generated_data": fmri_to_wandb_image(one_generated_sample_2d, title="Generated"),
                })

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
