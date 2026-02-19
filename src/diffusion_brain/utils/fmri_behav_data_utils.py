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


def preprocess_fmri_roi(args):
    """
    Extract ROI voxels from full fMRI data and save.
    Returns tensor [n_samples, n_roi_voxels]
    """
    print(f"Preprocessing fMRI ROI: {args.data.roi}", flush=True)
    
    # 1. Get ROI voxel indices
    roi_indices = get_roi_mask(args)
    print(f"ROI {args.data.roi}: {len(roi_indices)} voxels")
    
    # 2. Load full betas (memory-mapped)
    fmri_path = os.path.join(args.data.fmri_data_root, f"{args.data.subj}_{args.data.fmri_data_name}")
    full_betas = np.load(fmri_path, mmap_mode='r')  # [n_voxels, n_samples]
    
    # 3. Extract ROI voxels
    roi_betas = full_betas[roi_indices, :].T  # [n_samples, n_roi_voxels]
    roi_betas = torch.from_numpy(roi_betas.copy()).float()
    
    # 4. Save
    save_path = os.path.join(args.data.roi_defs_dir, f"roi_preselected", f"{args.data.roi_file}", f"{args.data.subj}_{args.data.roi}.pt")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(roi_betas, save_path)
    
    # Also save ROI indices for reference
    roi_indices_path = os.path.join(args.data.roi_defs_dir, f"roi_indices",  f"{args.data.roi_file}", f"{args.data.roi}.npy")
    if not os.path.exists(os.path.dirname(roi_indices_path)):
        os.makedirs(os.path.dirname(roi_indices_path))

    np.save(roi_indices_path, roi_indices)
    
    print(f"Saved ROI betas to {save_path}, shape: {roi_betas.shape}")
    
    return roi_betas, save_path


def ensure_fmri_roi_exists(args):
    """Thread-safe check and preprocessing for fMRI ROI data."""
    save_path = os.path.join(
        args.data.roi_defs_dir, f"roi_preselected", 
        f"{args.data.roi_file}",
        f"{args.data.subj}_{args.data.roi}.pt"
    )
    lock_path = save_path + ".lock"
    
    with FileLock(lock_path):
        if not os.path.isfile(save_path):
            print(f"fMRI ROI data not found at {save_path}. Preprocessing...", flush=True)
            preprocess_fmri_roi(args)
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