# Courtesy of: https://github.com/adriendoerig/visuo_llm/blob/main/src/nsd_visuo_semantics/utils/nsd_get_data_light.py

import os 
import numpy as np
import pandas as pd
import glob 
import re
import torch
import nibabel as nb
from filelock import FileLock
import rsatoolbox
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
    overall_cond_ann = np.load(os.path.join(args.data.behav_data_root, args.data.subj + "_all_conditions.npy"), allow_pickle=True) 
    
    test_cond_ann = np.load(os.path.join(args.data.behav_data_root, "common_515_indices.npy"), allow_pickle=True)
    train_cond_ann = np.array([i for i in overall_cond_ann if i not in test_cond_ann])

    train_pos_indices = np.where(np.isin(overall_cond_ann, train_cond_ann))[0]
    test_pos_indices = np.where(np.isin(overall_cond_ann, test_cond_ann))[0]
    
    return train_cond_ann, test_cond_ann, train_pos_indices, test_pos_indices


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
    
    if args.data.is_2d:
        roi_2d_data, locations = signal_to_2d(args)
        np.savez(save_path, data=roi_2d_data, locations=locations)
        print(f"Saved 2D ROI images to {save_path}, shape: {roi_2d_data.shape}", flush=True)
    else:            
        # 2. Load full betas (memory-mapped)
        fmri_path = os.path.join(args.data.fmri_data_root, f"{args.data.subj}_{args.data.fmri_data_name}")
        full_betas = np.load(fmri_path, mmap_mode='r')  # [n_voxels, n_samples]
        n_total_voxels = full_betas.shape[0]
        
        # 3. Extract ROI voxels
        roi_betas = full_betas[roi_indices, :].T  # [n_samples, n_roi_voxels]
        roi_betas = torch.from_numpy(roi_betas.copy()).float()
        
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
    # Determine save path based on 1D vs 2D
    # /roi_defs/roi_preselected_extended_2d_images_res_1.0/streams/subj01_5.ngz
    if args.data.is_2d:
        save_path = os.path.join(
            args.data.roi_defs_dir, f"roi_preselected_extended_2d_images_res_{args.data.grid_resolution_2d}", 
            f"{args.data.roi_file}",
            f"{args.data.subj}_{args.data.roi}.npz"
        )
    else:
        save_path = os.path.join(
            args.data.roi_defs_dir, f"roi_preselected", 
            f"{args.data.roi_file}",
            f"{args.data.subj}_{args.data.roi}.pt"
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
def signal_to_2d(args):
    grid_size = getattr(args.data, "grid_size_2d", None)
    grid_resolution = getattr(args.data, "grid_resolution_2d", 1.0)

    pts_left, _ = cortex.db.get_surf("fsaverage", "flat", hemisphere="left")
    pts_right, _ = cortex.db.get_surf("fsaverage", "flat", hemisphere="right")

    roi_indices = np.load(os.path.join(args.data.roi_defs_dir, f"roi_indices", f"{args.data.roi_file}", f"{args.data.roi}.npy"))
    data_roi = torch.load(os.path.join(args.data.roi_defs_dir, f"roi_preselected", f"{args.data.roi_file}", f"{args.data.subj}_{args.data.roi}.pt")).cpu().numpy()

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

    print(f"Grid size: {grid_size}x{grid_size}, Total points: {len(pts_roi)}, Unique grid points: {len(set(zip(x_grid, y_grid)))}")
    data_roi_2d = []

    
    for sample in range(data_roi.shape[0]):
        values = data_roi[sample] #10k values for each sample
        # Create matrix
        matrix_2d = np.full((grid_size, grid_size), np.nan, dtype=np.float32)

        matrix_2d[y_grid, x_grid] = values

        # Remove vertical NaN bands between hemispheres by cropping rows with all NaN
        valid_rows = ~np.all(np.isnan(matrix_2d), axis=0)
        if valid_rows.any():
            matrix_2d = matrix_2d[:, valid_rows]

        data_roi_2d.append(matrix_2d)

    locations_roi = np.where(data_roi_2d[0] != np.nan)

    return np.array(data_roi_2d)[:, None, :, :], locations_roi


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
