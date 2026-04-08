# Courtesy of: https://github.com/adriendoerig/visuo_llm/blob/main/src/nsd_visuo_semantics/utils/nsd_get_data_light.py

import os 
import numpy as np
import pandas as pd
import glob 
import re
import torch
import nibabel as nb
from filelock import FileLock
import cortex

import scipy
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import Subset

def read_behavior(behav_data_root, subject, session_index, trial_index=[]):
    """read_behavior [summary]

    Parameters
    ----------
    subject : str
        subject identifier, such as 'subj01'
    session_index : int
        which session, counting from 0
    trial_index : list, optional
        which trials from this session's behavior to return, by default [], which returns all trials

    Returns
    -------
    pandas DataFrame
        DataFrame containing the behavioral information for the requested trials
    """
    behavior_file = os.path.join(
        behav_data_root, f"{subject}", "behav", "responses.tsv"
    )

    behavior = pd.read_csv(behavior_file, delimiter="\t")
    print(behavior.head())
    # the behavior is encoded per run.
    # I'm now setting this function up so that it aligns with the timepoints in the fmri files,
    # i.e. using indexing per session, and not using the 'run' information.
    session_behavior = behavior[behavior["SESSION"] == session_index]

    if len(trial_index) == 0:
        trial_index = slice(0, len(session_behavior))

    return session_behavior.iloc[trial_index]

def get_conditions(behav_data_root, sub, n_sessions):
    """[summary]

    Args:
        behav_data_root ([type]): [description]
        sub ([type]): [description]
        n_sessions ([type]): [description]

    Returns:
        [type]: [description]
    """

    # read behaviour files for current subj
    conditions = []

    # loop over sessions
    for ses in range(n_sessions):
        ses_i = ses + 1
        print(f"\r\t\tsub: {sub} fetching condition trials in session: {ses_i}", end='')

        this_ses = np.asarray(read_behavior(behav_data_root, subject=sub, session_index=ses_i)["73KID"])

        # these are the 73K ids.
        valid_trials = [j for j, x in enumerate(this_ses)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:
            conditions.append(this_ses)

    return np.array(conditions)

def get_subject_conditions(behav_data_root, subj, n_sessions, keep_only_3repeats=True):

    # extract conditions data.
    # NOTES ABOUT HOW THIS WORKS:
    # get_conditions returns a list with one item for each session the subject attended. Each of these items contains
    # the NSD_ids for the images presented in that session. Then, we reshape all this into a single array, which now
    # contains all the NSD_ids for the subject, in the order in which they were shown. Next, we create a boolean list of
    # the same size as the conditions array, which assigns True to NSD_ids that are present 3x in the condition array.
    # We use this boolean to create conditions_sampled, which now contains all NSD_indices for stimuli the subject has
    # seen 3x. This list still contains the 3 repetitions of each stimulus, and is still in the stimulus presentation
    # order. For example: [46003, 61883,   829, ...]
    # Hence, we need to only keep each NSD_id once (since we compute everything on the average fMRI data over
    # the 3 presentations), and we also need to order them in increasing NSD_id order (so that we can then easily
    # for all subjects/models). Both of these desiderata are addressed by using np.unique (which sorts the unique idx).
    # So sample contains the unique NSD_ids for that subject, in increasing order (e.g. [ 14,  28,  72, ...]).
    # Importantly, the average betas loaded above are arranged in the same way, so that if we want to find the betas
    # for NSD_id=72, we just need to find the idx of 72 in sample (in the present example: 2). Using this method, we can
    # find the avg_betas corresponding to the shared 515 images as done below with subj_indices_515 (hint: the trick to
    # go from an ordered list of nsd_ids to finding the idx as described above is to use enumerate).
    # For example sample[subj_indices_515[0]] = conditions_515[0].

    # extract conditions data
    conditions = get_conditions(behav_data_root, subj, n_sessions)
    # we also need to reshape conditions to be ntrials x 1
    conditions = np.asarray(conditions).ravel()
    if keep_only_3repeats:
        # then we find the valid trials for which we do have 3 repetitions.
        conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
    else:
        conditions_bool = [True for x in conditions]
    # and identify those.
    conditions_sampled = conditions[conditions_bool]
    # find the subject's condition list (sample pool)
    # this sample is the same order as the betas
    sample = np.unique(conditions[conditions_bool])

    return conditions, conditions_sampled, sample


def get_train_test_indices(args):
    variant = getattr(args.data, "fmri_dataset_variant", "averaged")
    if variant == "unaveraged":
        return _get_train_test_indices_unaveraged(args)
    return _get_train_test_indices_averaged(args)


def _get_train_test_indices_averaged(args):
    overall_cond_ann = np.load(os.path.join(args.data.behav_data_root, args.data.subj + "_all_conditions.npy"), allow_pickle=True)

    test_cond_ann = np.load(os.path.join(args.data.behav_data_root, "common_515_indices.npy"), allow_pickle=True)
    train_cond_ann = np.array([i for i in overall_cond_ann if i not in test_cond_ann])

    train_pos_indices = np.where(np.isin(overall_cond_ann, train_cond_ann))[0]
    test_pos_indices = np.where(np.isin(overall_cond_ann, test_cond_ann))[0]

    return train_cond_ann, test_cond_ann, train_pos_indices, test_pos_indices


def _get_train_test_indices_unaveraged(args):
    """Train/test split for the unaveraged (full repetitions) NSD betas.

    The betas file has shape (n_trials, 327684) with a matching nsd_ids.npy
    of shape (n_trials,).  Multiple rows can share the same NSD image ID
    (one per repetition).  We split rows by whether their NSD ID belongs to
    the common-515 test set, keeping all repetitions on the correct side.

    Returns the same 4-tuple as the averaged variant:
        train_nsd_ids  – *unique* NSD IDs for ANN activation extraction
        test_nsd_ids   – *unique* NSD IDs for ANN activation extraction
        train_pos_indices – row indices into the betas file (may have repeats)
        test_pos_indices  – row indices into the betas file (may have repeats)
    """
    nsd_ids_path = os.path.join(args.data.fmri_data_root, args.data.subj, "nsd_ids.npy")
    all_nsd_ids = np.load(nsd_ids_path)  # (n_trials,)

    test_set = set(np.load(
        os.path.join(args.data.behav_data_root, "common_515_indices.npy"),
        allow_pickle=True,
    ).tolist())

    is_test = np.array([nsd_id in test_set for nsd_id in all_nsd_ids])
    test_pos_indices = np.where(is_test)[0]
    train_pos_indices = np.where(~is_test)[0]

    train_nsd_ids = np.unique(all_nsd_ids[train_pos_indices])
    test_nsd_ids = np.unique(all_nsd_ids[test_pos_indices])

    return train_nsd_ids, test_nsd_ids, train_pos_indices, test_pos_indices


ALL_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]


def get_multi_subject_train_test_indices(args):
    """Compute train/test indices for all 8 NSD subjects simultaneously.

    For the averaged variant:
      - Each subject's conditions are loaded from {subj}_all_conditions.npy.
      - Test set = common 515 images (same for every subject).
      - Train set = everything else per subject.

    Returns a dict with keys:
        per_subject: list of 8 dicts, each with:
            subj, train_nsd_ids, test_nsd_ids, train_indices, test_indices
        union_train_nsd_ids: sorted unique NSD IDs across all subjects' train sets
        union_test_nsd_ids: sorted unique NSD IDs across all subjects' test sets
                           (= common_515 for averaged variant)
    """
    variant = getattr(args.data, "fmri_dataset_variant", "averaged")
    subjects = getattr(args.data, "subjects", ALL_SUBJECTS)
    test_nsd_set = set(np.load(
        os.path.join(args.data.behav_data_root, "common_515_indices.npy"),
        allow_pickle=True,
    ).tolist())

    per_subject = []
    all_train_nsd = set()
    all_test_nsd = set()

    for subj in subjects:
        if variant == "unaveraged":
            nsd_ids_path = os.path.join(args.data.fmri_data_root, subj, "nsd_ids.npy")
            all_nsd_ids = np.load(nsd_ids_path)
            is_test = np.array([nsd_id in test_nsd_set for nsd_id in all_nsd_ids])
            test_indices = np.where(is_test)[0]
            train_indices = np.where(~is_test)[0]
            train_nsd_ids = np.unique(all_nsd_ids[train_indices])
            test_nsd_ids = np.unique(all_nsd_ids[test_indices])
        else:
            overall_cond = np.load(
                os.path.join(args.data.behav_data_root, f"{subj}_all_conditions.npy"),
                allow_pickle=True,
            )
            test_cond = np.array([c for c in overall_cond if c in test_nsd_set])
            train_cond = np.array([c for c in overall_cond if c not in test_nsd_set])
            train_indices = np.where(np.isin(overall_cond, train_cond))[0]
            test_indices = np.where(np.isin(overall_cond, test_cond))[0]
            train_nsd_ids = train_cond
            test_nsd_ids = test_cond

        per_subject.append({
            "subj": subj,
            "train_nsd_ids": train_nsd_ids,
            "test_nsd_ids": test_nsd_ids,
            "train_indices": train_indices,
            "test_indices": test_indices,
        })
        all_train_nsd.update(train_nsd_ids.tolist())
        all_test_nsd.update(test_nsd_ids.tolist())

    return {
        "per_subject": per_subject,
        "union_train_nsd_ids": np.sort(np.array(list(all_train_nsd))),
        "union_test_nsd_ids": np.sort(np.array(list(all_test_nsd))),
    }


def get_train_test_subsets(fmri_dataset, activations_dataset, args):
    train_cond_ann, test_cond_ann, train_pos_indices, test_pos_indices = get_train_test_indices(args)

    train_fmri_dataset = Subset(fmri_dataset, train_pos_indices)
    test_fmri_dataset = Subset(fmri_dataset, test_pos_indices)
    
    train_activations_dataset = Subset(activations_dataset, train_cond_ann)
    test_activations_dataset = Subset(activations_dataset, test_cond_ann)

    return train_fmri_dataset, test_fmri_dataset, train_activations_dataset, test_activations_dataset


################### fmri utils ########################
def get_roi_mask(args):
    """Load ROI mask and return voxel indices."""
    roi_defs_dir = args.data.roi_defs_dir
    roi_file = args.data.roi_file
    roi = args.data.roi
    
    # Load left and right hemisphere masks
    try:
        lh_file = os.path.join(roi_defs_dir, f"lh.{roi_file}.mgz")
        rh_file = os.path.join(roi_defs_dir, f"rh.{roi_file}.mgz")
        maskdata_lh = nb.load(lh_file).get_fdata().squeeze()
        maskdata_rh = nb.load(rh_file).get_fdata().squeeze()
    except FileNotFoundError:
        lh_file = os.path.join(roi_defs_dir, f"lh.{roi_file}.npy")
        rh_file = os.path.join(roi_defs_dir, f"rh.{roi_file}.npy")
        maskdata_lh = np.load(lh_file)
        maskdata_rh = np.load(rh_file)
    
    maskdata = np.hstack((maskdata_lh, maskdata_rh))
    roi_indices = np.where(maskdata == roi)[0]
    
    return roi_indices


def _load_full_betas(args):
    """Load full betas array (memory-mapped) and return (array, is_samples_first).

    Averaged variant: shape (n_voxels, n_samples) — stored as {subj}_{filename}
    Unaveraged variant: shape (n_samples, n_voxels) — stored as {subj}/{filename}
    """
    variant = getattr(args.data, "fmri_dataset_variant", "averaged")
    if variant == "unaveraged":
        fmri_path = os.path.join(
            args.data.fmri_data_root, args.data.subj, args.data.fmri_data_name,
        )
        full_betas = np.load(fmri_path, mmap_mode='r')  # (n_samples, n_voxels)
        return full_betas, True
    else:
        fmri_path = os.path.join(
            args.data.fmri_data_root,
            f"{args.data.subj}_{args.data.fmri_data_name}",
        )
        full_betas = np.load(fmri_path, mmap_mode='r')  # (n_voxels, n_samples)
        return full_betas, False


def preprocess_fmri_roi(args, save_path=None):
    """
    Extract ROI voxels from full fMRI data and save.
    Handles both 1D and 2D (transformed) formats based on args.data.is_2d.
    For 2D mode: converts all samples to 2D images and saves them pre-processed.
    Returns tensor [n_samples, n_roi_voxels] for 1D or 2D grid representation.
    """
    print(f"Preprocessing fMRI ROI: {args.data.roi}", flush=True)
    # 1. Get ROI voxel indices
    roi_indices = get_roi_mask(args)
    print(f"ROI {args.data.roi}: {len(roi_indices)} voxels")

    # Load full betas (memory-mapped)
    full_betas, samples_first = _load_full_betas(args)

    # Extract ROI voxels → [n_samples, n_roi_voxels]
    if samples_first:
        roi_betas = full_betas[:, roi_indices]
    else:
        roi_betas = full_betas[roi_indices, :].T

    if args.data.is_2d:
        # Pass pre-extracted ROI data directly to avoid loading from a 1D cache
        roi_2d_data, locations = signal_to_2d(args, data_roi=roi_betas)
        np.savez(save_path, data=roi_2d_data, locations=locations)
        print(f"Saved 2D ROI images to {save_path}, shape: {roi_2d_data.shape}", flush=True)
    else:
        roi_betas = torch.from_numpy(np.array(roi_betas).copy()).float()
        torch.save(roi_betas, save_path)
        print(f"Saved 1D ROI betas to {save_path}, shape: {roi_betas.shape}", flush=True)

    # Also save ROI indices for reference
    roi_indices_dir = os.path.join(args.data.roi_defs_dir, f"roi_indices",  f"{args.data.roi_file}")
    if not os.path.exists(roi_indices_dir):
        os.makedirs(roi_indices_dir)
    roi_indices_path = os.path.join(roi_indices_dir, f"{args.data.roi}.npy")
    np.save(roi_indices_path, roi_indices)
        
   # return roi_betas, save_path


def ensure_fmri_roi_exists(args):
    """Thread-safe check and preprocessing for fMRI ROI data.

    Determines the appropriate save path based on 1D vs 2D configuration,
    then delegates to preprocess_fmri_roi() which handles both formats.
    """
    variant = getattr(args.data, "fmri_dataset_variant", "averaged")
    variant_suffix = "_unaveraged" if variant == "unaveraged" else ""

    # Determine save path based on 1D vs 2D
    # /roi_defs/roi_preselected_extended_2d_images_res_1.0/streams/subj01_5.ngz
    if args.data.is_2d:
        save_path = os.path.join(
            args.data.roi_defs_dir, f"roi_preselected_extended_2d_images_res_{args.data.grid_resolution_2d}",
            f"{args.data.roi_file}",
            f"{args.data.subj}_{args.data.roi}{variant_suffix}.npz"
        )
    else:
        save_path = os.path.join(
            args.data.roi_defs_dir, f"roi_preselected",
            f"{args.data.roi_file}",
            f"{args.data.subj}_{args.data.roi}{variant_suffix}.pt"
        )

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))    
        
    lock_path = save_path + ".lock"
    
    with FileLock(lock_path):
        if not os.path.isfile(save_path):
            print(f"fMRI ROI data not found at {save_path}. Preprocessing...", flush=True)
            preprocess_fmri_roi(args, save_path)
        else:
            print(f"fMRI ROI data found at {save_path}", flush=True)
    
    return save_path

################ RDM manipulations ########################
def compute_rdm(data, args, regime="train", method='correlation'):
    train_nsd_ids, test_nsd_ids, _, _ = get_train_test_indices(args)

    nsd_ids = train_nsd_ids if regime == 'train' else test_nsd_ids
    obs_descriptors = {'conds': [f'stim_{i}' for i in nsd_ids]}

    # 3. Create the rsatoolbox Dataset object
    dataset = rsatoolbox.data.Dataset(
        measurements=data,
        obs_descriptors=obs_descriptors
    )
    rdm = rsatoolbox.rdm.calc_rdm(dataset, method=method)
    return rdm

########################## 1D -> 2D utils ##########################
def signal_to_2d(args, **kwargs):
    grid_size = getattr(args.data, "grid_size_2d", None)
    grid_resolution = getattr(args.data, "grid_resolution_2d", 1.0)

    pts_left, _ = cortex.db.get_surf("fsaverage", "flat", hemisphere="left")
    pts_right, _ = cortex.db.get_surf("fsaverage", "flat", hemisphere="right")

    roi_indices = np.load(os.path.join(args.data.roi_defs_dir, f"roi_indices", f"{args.data.roi_file}", f"{args.data.roi}.npy"))
    
    # if we work with one signal, a pre-loaded array, or the whole dataset:
    if kwargs.get("one_signal_to_transform") is not None:
        data_roi = kwargs["one_signal_to_transform"]
    elif kwargs.get("data_roi") is not None:
        data_roi = kwargs["data_roi"]
        if isinstance(data_roi, torch.Tensor):
            data_roi = data_roi.cpu().numpy()
    else:
        data_roi = torch.load(os.path.join(args.data.roi_defs_dir, f"roi_preselected", f"{args.data.roi_file}", f"{args.data.subj}_{args.data.roi}.pt")).cpu().numpy()

    # Ensure data_roi is 2D: (num_samples, num_voxels)
    if data_roi.ndim == 1:
        data_roi = data_roi[np.newaxis, :]

    pts_general = np.concatenate([pts_left[:,:2], pts_right[:,:2]], axis=0)
    pts_roi = pts_general[roi_indices]

    # Create 2D matrix from scatter coordinates and values
    x = pts_roi[:, 0]
    y = pts_roi[:, 1]

    # Calculate grid_size from resolution if not provided
    if grid_size is None:
        # grid_resolution = (range / grid_size)
        # Solve for grid_size: grid_size = range / resolution
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        grid_size = int(max(x_range, y_range) / grid_resolution) + 1

    # Normalize to grid
    x_grid = ((x - x.min()) / (x.max() - x.min() + 1e-8) * (grid_size - 1)).astype(int)
    y_grid = ((y - y.min()) / (y.max() - y.min() + 1e-8) * (grid_size - 1)).astype(int)

    n_unique = len(set(zip(x_grid, y_grid)))
    n_collisions = len(pts_roi) - n_unique
    print(f"Grid size: {grid_size}x{grid_size}, Total points: {len(pts_roi)}, "
          f"Unique grid points: {n_unique}, Collisions: {n_collisions}")

    # Precompute per-pixel count for averaging colliding vertices.
    # np.add.at is unbuffered so it correctly accumulates duplicates.
    count_2d = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(count_2d, (y_grid, x_grid), 1)

    # Boolean mask of active pixels (>0 vertices mapped here) — computed
    # once from the grid mapping, independent of any sample's values.
    active_mask = count_2d > 0

    data_roi_2d = []
    for sample in range(data_roi.shape[0]):
        values = data_roi[sample]
        matrix_2d = np.zeros((grid_size, grid_size), dtype=np.float32)
        np.add.at(matrix_2d, (y_grid, x_grid), values)
        matrix_2d[active_mask] /= count_2d[active_mask]

        # Remove all-zero columns (gap between hemispheres)
        valid_cols = ~np.all(matrix_2d == 0, axis=0)
        if valid_cols.any():
            matrix_2d = matrix_2d[:, valid_cols]

        data_roi_2d.append(matrix_2d)

    # Active pixel locations from the geometry, not from sample values
    # (avoids missing pixels where sample 0 happens to be exactly zero)
    valid_cols_mask = ~np.all(count_2d == 0, axis=0)
    cropped_active = active_mask[:, valid_cols_mask]
    locations_roi = np.where(cropped_active)

    return np.array(data_roi_2d)[:, None, :, :], locations_roi

def get_roi_flatmap_crops_compact(roi_names, subject='fsaverage', height=1024, padding=5, combined=True):
    """
    Pre-compute tight 2D flatmap crops for each ROI, one per hemisphere.
    Removes unnecessary NaN padding between hemispheres.
    """
    roi_indices = np.load("/home/belova/Desktop/diffusion_brain_alignment/src/diffusion_brain/data/roi_defs/roi_indices/streams/5.npy")
    roi_dict = {"5": roi_indices}

    n_total = cortex.db.get_surf(subject, 'flat', hemisphere='left')[0].shape[0] + \
              cortex.db.get_surf(subject, 'flat', hemisphere='right')[0].shape[0]

    # Get vertex-to-pixel mapping
    dummy = cortex.Vertex(np.arange(n_total, dtype=np.float32), subject=subject)
    im, _ = cortex.quickflat.make_flatmap_image(dummy, height=height)
    vertex_map = im[:, :, 0] if im.ndim == 3 else im
    
    n_left = cortex.db.get_surf(subject, 'flat', hemisphere='left')[0].shape[0]
    W_full = vertex_map.shape[1]
    lh_region = (0, W_full // 2)
    rh_region = (W_full // 2, W_full)

    crops = {}
    meta = {}

    for roi in roi_names:
        roi_verts = set(roi_dict[roi].astype(int))
        roi_pixel_mask = np.zeros(vertex_map.shape, dtype=bool)
        valid_pixels = ~np.isnan(vertex_map)
        flat_indices = vertex_map[valid_pixels].astype(int)
        belongs = np.array([v in roi_verts for v in flat_indices])
        temp_mask = np.zeros(vertex_map.shape, dtype=bool)
        temp_mask[valid_pixels] = belongs
        roi_pixel_mask = temp_mask

        crops[roi] = {}
        meta[roi] = {}

        for hemi, (c_start, c_end) in [('lh', lh_region), ('rh', rh_region)]:
            hemi_mask = roi_pixel_mask.copy()
            hemi_mask[:, :c_start] = False
            hemi_mask[:, c_end:] = False

            rows = np.where(hemi_mask.any(axis=1))[0]
            cols = np.where(hemi_mask.any(axis=0))[0]

            if len(rows) == 0 or len(cols) == 0:
                crops[roi][hemi] = None
                meta[roi][hemi] = None
                continue

            r0 = max(0, rows.min() - padding)
            r1 = min(vertex_map.shape[0], rows.max() + padding + 1)
            c0 = max(c_start, cols.min() - padding)
            c1 = min(c_end, cols.max() + padding + 1)

            bbox = (r0, r1, c0, c1)
            crops[roi][hemi] = np.array(bbox)
            meta[roi][hemi] = {
                'bbox': bbox,
                'mask': hemi_mask[r0:r1, c0:c1],
                'vertex_map_crop': vertex_map[r0:r1, c0:c1],
                'hemi': hemi
            }

    if combined:
        lh_bbox = meta[roi]['lh']['bbox']
        rh_bbox = meta[roi]['rh']['bbox']
        gap = 2
        meta[roi]['combined_info'] = {
            'lh_bbox': lh_bbox,
            'rh_bbox': rh_bbox,
            'gap': gap
        }

    return crops, meta, vertex_map, n_left


def signal_to_flatmap_crop_compact(signal_1d, subject, roi, meta, vertex_map, combined=True):
    """Forward: 1D fsaverage signal → compact 2D flatmap (minimal padding)."""
    if signal_1d.shape[0] != 327684:
        raise ValueError(f"Signal length {signal_1d.shape[0]} != 327684")
    
    vertex_data = cortex.Vertex(signal_1d.astype(np.float32), subject=subject)
    im, _ = cortex.quickflat.make_flatmap_image(vertex_data)
    flatmap = im[:, :, 0] if im.ndim == 3 else im

    if combined:
        lh_bbox = meta[roi]['lh']['bbox']
        rh_bbox = meta[roi]['rh']['bbox']
        gap = meta[roi]['combined_info']['gap']
        
        r0_lh, r1_lh, c0_lh, c1_lh = lh_bbox
        r0_rh, r1_rh, c0_rh, c1_rh = rh_bbox
        
        lh_crop = flatmap[r0_lh:r1_lh, c0_lh:c1_lh]
        rh_crop = flatmap[r0_rh:r1_rh, c0_rh:c1_rh]
        
        h_max = max(lh_crop.shape[0], rh_crop.shape[0])
        
        lh_padded = np.full((h_max, lh_crop.shape[1]), np.nan, dtype=np.float32)
        rh_padded = np.full((h_max, rh_crop.shape[1]), np.nan, dtype=np.float32)
        
        lh_padded[:lh_crop.shape[0]] = lh_crop
        rh_padded[:rh_crop.shape[0]] = rh_crop
        
        gap_array = np.full((h_max, gap), np.nan, dtype=np.float32)
        
        result = np.concatenate([lh_padded, gap_array, rh_padded], axis=1)
        return result.astype(np.float32)


def flatmap_crop_compact_to_signal_1d(crop_2d, subject, roi, meta, combined=True):
    """Inverse: compact 2D flatmap → 1D fsaverage signal (lossless)."""
    signal_1d = np.zeros(327684, dtype=np.float32)
    
    if combined:
        lh_bbox = meta[roi]['lh']['bbox']
        rh_bbox = meta[roi]['rh']['bbox']
        gap = meta[roi]['combined_info']['gap']
        
        r0_lh, r1_lh, c0_lh, c1_lh = lh_bbox
        r0_rh, r1_rh, c0_rh, c1_rh = rh_bbox
        
        lh_height = r1_lh - r0_lh
        lh_width = c1_lh - c0_lh
        rh_width = c1_rh - c0_rh
        
        lh_crop = crop_2d[:lh_height, :lh_width]
        rh_start_col = lh_width + gap
        rh_crop = crop_2d[:, rh_start_col:rh_start_col + rh_width]
        
        # Reconstruct LH
        vertex_map_lh = meta[roi]['lh']['vertex_map_crop']
        valid_pixels = ~np.isnan(vertex_map_lh)
        flat_vertices = vertex_map_lh[valid_pixels].astype(int)
        flat_values = lh_crop[valid_pixels]
        signal_1d[flat_vertices] = flat_values
        
        # Reconstruct RH
        vertex_map_rh = meta[roi]['rh']['vertex_map_crop']
        valid_pixels = ~np.isnan(vertex_map_rh)
        flat_vertices = vertex_map_rh[valid_pixels].astype(int)
        flat_values = rh_crop[valid_pixels]
        signal_1d[flat_vertices] = flat_values
    
    return signal_1d

# def compute_grid_no_collisions(xy_roi):
#     """Map ROI vertices to 2D grid without collisions."""
#     x, y = xy_roi[:, 0], xy_roi[:, 1]
#     n = len(xy_roi)

#     n_rows = int(np.ceil(np.sqrt(n)))

#     y_bins = np.round(
#         (y - y.min()) /
#         (y.max() - y.min() + 1e-8) * (n_rows - 1)
#     ).astype(int)

#     x_idx = np.zeros(n, dtype=int)
#     y_idx = y_bins.copy()

#     for row in range(n_rows):
#         mask = y_bins == row
#         if mask.sum() == 0:
#             continue
#         col_order = np.argsort(x[mask])
#         col_indices = np.empty_like(col_order)
#         col_indices[col_order] = np.arange(mask.sum())
#         x_idx[mask] = col_indices

#     H = y_idx.max() + 1
#     W = x_idx.max() + 1

#     positions = np.stack([y_idx, x_idx], axis=1)
#     n_unique = len(np.unique(positions, axis=0))
#     assert n_unique == n, f"Collisions detected: {n - n_unique}"

#     return x_idx, y_idx, H, W


# def precompute_roi_indices(roi_verts, args):
#     """
#     Precompute 2D grid indices for ROI vertices on fsaverage surface.
#     Handles left and right hemispheres independently, then combines them.
#     """
#     print("Precomuting the extended 2D image indices for the ROI vertices...", flush=True)
#     pts_left, _ = cortex.db.get_surf('fsaverage', 'flat', hemisphere='left')
#     pts_right, _ = cortex.db.get_surf('fsaverage', 'flat', hemisphere='right')

#     xy_left = pts_left[:, :2]
#     xy_right = pts_right[:, :2]
#     n_left = len(pts_left)

#     roi_verts = roi_verts.astype(int)
#     lh_mask = roi_verts < n_left
#     rh_mask = ~lh_mask

#     lh_verts_global = roi_verts[lh_mask]
#     rh_verts_global = roi_verts[rh_mask]
#     rh_verts_local = rh_verts_global - n_left

#     result = {}
#     for hemi, xy, verts_global, verts_local in [
#         ('lh', xy_left, lh_verts_global, lh_verts_global),
#         ('rh', xy_right, rh_verts_global, rh_verts_local),
#     ]:
#         if len(verts_global) == 0:
#             result[hemi] = None
#             continue

#         xy_roi = xy[verts_local]
#         x_idx, y_idx, H, W = compute_grid_no_collisions(xy_roi)

#         result[hemi] = {
#             'verts_global': verts_global,
#             'verts_local': verts_local,
#             'x_idx': x_idx,
#             'y_idx': y_idx,
#             'H': H,
#             'W': W,
#         }

#     lh_info = result['lh']
#     rh_info = result['rh']

#     gap = 5
#     combined_H = max(lh_info['H'], rh_info['H'])
#     combined_W = lh_info['W'] + gap + rh_info['W']
#     rh_x_idx_offset = rh_info['x_idx'] + lh_info['W'] + gap

#     output = {
#         'lh': {
#             'verts_global': lh_info['verts_global'],
#             'verts_local': lh_info['verts_local'],
#             'x_idx': lh_info['x_idx'],
#             'y_idx': lh_info['y_idx'],
#             'H': lh_info['H'],
#             'W': lh_info['W'],
#         },
#         'rh': {
#             'verts_global': rh_info['verts_global'],
#             'verts_local': rh_info['verts_local'],
#             'x_idx': rh_x_idx_offset,
#             'y_idx': rh_info['y_idx'],
#             'H': rh_info['H'],
#             'W': rh_info['W'],
#         },
#         'H': combined_H,
#         'W': combined_W,
#         'rh_x_idx_local': rh_info['x_idx'],
#         'rh_y_idx_local': rh_info['y_idx'],
#     }
    
#     save_path = os.path.join(
#         args.data.roi_defs_dir, f"roi_preselected_extended_2d_images_info", 
#         f"{args.data.roi_file}",
#         f"{args.data.subj}_{args.data.roi}.pt"
#     )
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     torch.save(output, save_path)

#     return output


# def signal_to_2d(signal_1d, roi_info, combined=True):
#     """Convert 1D fsaverage signal to 2D matrix."""
#     lh = roi_info['lh']
#     rh = roi_info['rh']

#     if combined:
#         matrix = np.full((roi_info['H'], roi_info['W']),0, dtype=np.float32)
#         matrix[lh['y_idx'], lh['x_idx']] = signal_1d[lh['verts_global']]
#         matrix[rh['y_idx'], rh['x_idx']] = signal_1d[rh['verts_global']]
#         return matrix
#     else:
#         lh_matrix = np.full((lh['H'], lh['W']),0, dtype=np.float32)
#         rh_matrix = np.full((rh['H'], rh['W']),0, dtype=np.float32)
#         lh_matrix[lh['y_idx'], lh['x_idx']] = signal_1d[lh['verts_global']]
#         rh_matrix[roi_info['rh_y_idx_local'], roi_info['rh_x_idx_local']] = signal_1d[rh['verts_global']]
#         return lh_matrix, rh_matrix


# def matrix_to_signal_1d(matrix_or_tuple, n_total, roi_info, combined=True):
#     """Convert 2D matrix back to 1D fsaverage signal."""
#     signal = np.full(n_total, 0, dtype=np.float32)
#     lh = roi_info['lh']
#     rh = roi_info['rh']

#     if combined:
#         matrix = matrix_or_tuple
#         signal[lh['verts_global']] = matrix[lh['y_idx'], lh['x_idx']]
#         signal[rh['verts_global']] = matrix[rh['y_idx'], rh['x_idx']]
#     else:
#         lh_matrix, rh_matrix = matrix_or_tuple
#         signal[lh['verts_global']] = lh_matrix[lh['y_idx'], lh['x_idx']]
#         signal[rh['verts_global']] = rh_matrix[roi_info['rh_y_idx_local'], roi_info['rh_x_idx_local']]
#     return signal
