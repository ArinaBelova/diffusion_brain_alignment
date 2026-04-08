# TODO: write a dataloader that takes as inputs coco and nifti files and outputs paired data for training
# where data is in the form of (image, annotation) where image is a 2D slice from the nifti file 
# and annotation is the corresponding activations from layer of ANN, given the coco image

from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import numpy as np
import os
import tempfile

from diffusion_brain.datasets.paired_brain_ann_dataset import AnnActivationsDataset, PairedBrainAnnDataset
from diffusion_brain.utils.ann_activations_utils import ensure_activations_exist
from diffusion_brain.utils.fmri_behav_data_utils import get_train_test_subsets, get_train_test_indices, ensure_fmri_roi_exists, get_multi_subject_train_test_indices

class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        print("index to query from dataloader: ", index)
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return len(self.datasets[0])

# def get_ann_brain_dataloader(args):
#     fmri_dataset = FmriDataset(args)
#     activations_dataset = AnnActivationsDataset(args)

#     train_fmri_dataset, test_fmri_dataset, train_activations_dataset, test_activations_dataset = get_train_test_subsets(fmri_dataset, activations_dataset, args)

#     train_dataset = ZipDataset(train_fmri_dataset, train_activations_dataset) # instead of TensorDataset, which throws an error
#     test_dataset = ZipDataset(test_fmri_dataset, test_activations_dataset)

#     train_dataloader = DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers, drop_last=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.validation.batch_size, shuffle=False, num_workers=args.validation.num_workers, drop_last=False)

#     return train_dataloader, test_dataloader

def _build_act_index_map(nsd_ids_per_row, unique_nsd_ids):
    """Build a mapping from each fMRI row to its index in the unique activations array.

    For the averaged variant this is identity (1:1).
    For the unaveraged variant, multiple fMRI rows share the same NSD image ID,
    so they all point to the same activation index.

    Args:
        nsd_ids_per_row: NSD IDs for each fMRI row in this split (may have repeats)
        unique_nsd_ids: sorted unique NSD IDs used for activation extraction

    Returns:
        np.ndarray of shape (len(nsd_ids_per_row),) mapping row → activation index
    """
    id_to_idx = {int(nsd_id): i for i, nsd_id in enumerate(unique_nsd_ids)}
    return np.array([id_to_idx[int(nsd_id)] for nsd_id in nsd_ids_per_row])


def get_ann_brain_dataloader(args):
    train_nsd_ids, test_nsd_ids, train_indices, test_indices = get_train_test_indices(args)

    # 2. Ensure fMRI ROI data exists (main process, thread-safe)
    fmri_roi_path = ensure_fmri_roi_exists(args)

    # 3. Ensure ANN activations exist (main process, thread-safe)
    train_activations_path = os.path.join(
        args.data.ann_activations_data_path,
        args.data.ann_model,
        f"activations_weights_{args.data.ann_model_weights}_layer_{args.data.layer_name}_{len(train_nsd_ids)}_samples.pt"
    )
    test_activations_path = os.path.join(
        args.data.ann_activations_data_path,
        args.data.ann_model,
        f"activations_weights_{args.data.ann_model_weights}_layer_{args.data.layer_name}_{len(test_nsd_ids)}_samples.pt"
    )
    ensure_activations_exist(train_activations_path, train_nsd_ids, args)
    ensure_activations_exist(test_activations_path, test_nsd_ids, args)

    # 4. For unaveraged variant, build the mapping from fMRI rows to activation indices
    #    (multiple fMRI repetitions share the same ANN activation).
    #    For averaged variant this is None → dataset uses identity mapping.
    variant = getattr(args.data, "fmri_dataset_variant", "averaged")
    train_act_index = None
    test_act_index = None
    if variant == "unaveraged":
        all_nsd_ids = np.load(
            os.path.join(args.data.fmri_data_root, args.data.subj, "nsd_ids.npy")
        )
        train_act_index = _build_act_index_map(all_nsd_ids[train_indices], train_nsd_ids)
        test_act_index = _build_act_index_map(all_nsd_ids[test_indices], test_nsd_ids)

    # 5. Create dataset (train first, then pass its stats to test to avoid data leakage)
    fmri_norm_mode = getattr(args.data, "fmri_norm_mode", "active_std")

    train_dataset = PairedBrainAnnDataset(
        activations_path=train_activations_path,
        fmri_roi_path=fmri_roi_path,
        sample_indices=train_indices,
        is_2d=args.data.is_2d,
        act_index_map=train_act_index,
        fmri_norm_mode=fmri_norm_mode,
    )

    test_dataset = PairedBrainAnnDataset(
        activations_path=test_activations_path,
        fmri_roi_path=fmri_roi_path,
        sample_indices=test_indices,
        is_2d=args.data.is_2d,
        act_mean=train_dataset.act_mean,
        act_std=train_dataset.act_std,
        fmri_scale=getattr(train_dataset, 'fmri_scale', None),
        act_index_map=test_act_index,
        fmri_norm_mode=fmri_norm_mode,
    )

    # 6. DataLoader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.validation.batch_size,
        shuffle=False,
        num_workers=args.validation.num_workers,
        drop_last=True
    )

    if args.state != "train":
        return test_dataloader

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers,
        drop_last=True
    )

    return train_dataloader, test_dataloader


def get_ann_brain_dataloader_multisubject(args):
    """Build dataloaders that concatenate data from all NSD subjects.

    Train set: union of all subjects' train partitions (~9485 × 8 samples).
    Test set: 515 common images × 8 subjects (per-subject evaluation done downstream).

    ANN activations are extracted once for the union of NSD IDs across subjects,
    and each fMRI row points into that shared activation tensor via act_index_map.

    Returns the same types as get_ann_brain_dataloader:
        (train_dataloader, test_dataloader) when args.state == "train"
        test_dataloader when args.state != "train"
    """
    variant = getattr(args.data, "fmri_dataset_variant", "averaged")
    fmri_norm_mode = getattr(args.data, "fmri_norm_mode", "active_std")

    # 1. Get per-subject indices and union NSD IDs
    ms_info = get_multi_subject_train_test_indices(args)
    per_subject = ms_info["per_subject"]
    union_train_nsd = ms_info["union_train_nsd_ids"]
    union_test_nsd = ms_info["union_test_nsd_ids"]

    print(f"Multi-subject: {len(per_subject)} subjects, "
          f"union train NSD IDs: {len(union_train_nsd)}, "
          f"union test NSD IDs: {len(union_test_nsd)}", flush=True)

    # 2. Ensure ANN activations exist for the union sets
    train_activations_path = os.path.join(
        args.data.ann_activations_data_path,
        args.data.ann_model,
        f"activations_weights_{args.data.ann_model_weights}_layer_{args.data.layer_name}_{len(union_train_nsd)}_samples.pt"
    )
    test_activations_path = os.path.join(
        args.data.ann_activations_data_path,
        args.data.ann_model,
        f"activations_weights_{args.data.ann_model_weights}_layer_{args.data.layer_name}_{len(union_test_nsd)}_samples.pt"
    )
    ensure_activations_exist(train_activations_path, union_train_nsd, args)
    ensure_activations_exist(test_activations_path, union_test_nsd, args)

    # Precompute NSD ID → index lookups for union arrays
    train_id_to_idx = {int(nsd_id): i for i, nsd_id in enumerate(union_train_nsd)}
    test_id_to_idx = {int(nsd_id): i for i, nsd_id in enumerate(union_test_nsd)}

    # 3. Load fMRI data from all subjects into memory-mapped files to avoid OOM.
    #    First pass: collect sizes and metadata. Second pass: write into memmap.
    all_train_subject_ids = []
    all_test_subject_ids = []
    all_train_act_idx = []
    all_test_act_idx = []

    original_subj = args.data.subj  # save to restore later

    # First pass: determine shapes and collect metadata
    subj_train_sizes = []
    subj_test_sizes = []
    sample_shape = None  # will be determined from first subject

    for subj_idx, info in enumerate(per_subject):
        subj = info["subj"]
        args.data.subj = subj
        fmri_roi_path = ensure_fmri_roi_exists(args)

        # Peek at shape without loading full data (for 2D, load one subject to get spatial dims)
        if sample_shape is None:
            if args.data.is_2d:
                fmri_peek = np.load(fmri_roi_path, allow_pickle=True)["data"]
                sample_shape = fmri_peek.shape[1:]  # (1, H, W) or (n_voxels,)
                sample_dtype = fmri_peek.dtype
                del fmri_peek
            else:
                fmri_peek = torch.load(fmri_roi_path, map_location="cpu")
                if isinstance(fmri_peek, torch.Tensor):
                    fmri_peek = fmri_peek.numpy()
                sample_shape = fmri_peek.shape[1:]
                sample_dtype = fmri_peek.dtype
                del fmri_peek

        n_train = len(info["train_indices"])
        n_test = len(info["test_indices"])
        subj_train_sizes.append(n_train)
        subj_test_sizes.append(n_test)
        all_train_subject_ids.extend([subj_idx] * n_train)
        all_test_subject_ids.extend([subj_idx] * n_test)

        # Build act_index_map
        if variant == "unaveraged":
            nsd_ids_all = np.load(
                os.path.join(args.data.fmri_data_root, subj, "nsd_ids.npy")
            )
            train_row_nsd = nsd_ids_all[info["train_indices"]]
            test_row_nsd = nsd_ids_all[info["test_indices"]]
        else:
            train_row_nsd = info["train_nsd_ids"]
            test_row_nsd = info["test_nsd_ids"]

        all_train_act_idx.extend([train_id_to_idx[int(nid)] for nid in train_row_nsd])
        all_test_act_idx.extend([test_id_to_idx[int(nid)] for nid in test_row_nsd])

        print(f"  {subj} (id={subj_idx}): train={n_train}, test={n_test}", flush=True)

    total_train = sum(subj_train_sizes)
    total_test = sum(subj_test_sizes)

    # Create memory-mapped files for concatenated fMRI data
    # Use LOCAL_JOB_DIR (fast local scratch) if available, else tempdir
    scratch_dir = os.environ.get("LOCAL_JOB_DIR", tempfile.gettempdir())
    train_mmap_path = os.path.join(scratch_dir, "concat_train_fmri.npy")
    test_mmap_path = os.path.join(scratch_dir, "concat_test_fmri.npy")

    concat_train_fmri = np.lib.format.open_memmap(
        train_mmap_path, mode='w+', dtype=sample_dtype,
        shape=(total_train, *sample_shape),
    )
    concat_test_fmri = np.lib.format.open_memmap(
        test_mmap_path, mode='w+', dtype=sample_dtype,
        shape=(total_test, *sample_shape),
    )

    # Second pass: load each subject and write directly into memmap
    train_offset = 0
    test_offset = 0
    for subj_idx, info in enumerate(per_subject):
        subj = info["subj"]
        args.data.subj = subj
        fmri_roi_path = ensure_fmri_roi_exists(args)

        if args.data.is_2d:
            fmri_all = np.load(fmri_roi_path, allow_pickle=True)["data"]
        else:
            fmri_all = torch.load(fmri_roi_path, map_location="cpu")
            if isinstance(fmri_all, torch.Tensor):
                fmri_all = fmri_all.numpy()

        n_train = subj_train_sizes[subj_idx]
        n_test = subj_test_sizes[subj_idx]
        concat_train_fmri[train_offset:train_offset + n_train] = fmri_all[info["train_indices"]]
        concat_test_fmri[test_offset:test_offset + n_test] = fmri_all[info["test_indices"]]
        train_offset += n_train
        test_offset += n_test
        del fmri_all  # free immediately

    concat_train_fmri.flush()
    concat_test_fmri.flush()

    args.data.subj = original_subj  # restore

    train_subject_ids = np.array(all_train_subject_ids, dtype=np.int64)
    test_subject_ids = np.array(all_test_subject_ids, dtype=np.int64)
    train_act_index = np.array(all_train_act_idx, dtype=np.int64)
    test_act_index = np.array(all_test_act_idx, dtype=np.int64)

    print(f"Concatenated train fMRI: {concat_train_fmri.shape}, "
          f"test fMRI: {concat_test_fmri.shape} (memory-mapped)", flush=True)

    # 4. Create datasets (train first for normalisation stats)
    train_dataset = PairedBrainAnnDataset(
        activations_path=train_activations_path,
        fmri_roi_path=None,
        sample_indices=None,
        is_2d=args.data.is_2d,
        act_index_map=train_act_index,
        fmri_norm_mode=fmri_norm_mode,
        subject_ids=train_subject_ids,
        fmri_data_preloaded=concat_train_fmri,
    )

    test_dataset = PairedBrainAnnDataset(
        activations_path=test_activations_path,
        fmri_roi_path=None,
        sample_indices=None,
        is_2d=args.data.is_2d,
        act_mean=train_dataset.act_mean,
        act_std=train_dataset.act_std,
        fmri_scale=getattr(train_dataset, 'fmri_scale', None),
        act_index_map=test_act_index,
        fmri_norm_mode=fmri_norm_mode,
        subject_ids=test_subject_ids,
        fmri_data_preloaded=concat_test_fmri,
    )

    # 5. DataLoaders
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.validation.batch_size,
        shuffle=False,
        num_workers=args.validation.num_workers,
        drop_last=True,
    )

    if args.state != "train":
        return test_dataloader

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers,
        drop_last=True,
    )

    return train_dataloader, test_dataloader
