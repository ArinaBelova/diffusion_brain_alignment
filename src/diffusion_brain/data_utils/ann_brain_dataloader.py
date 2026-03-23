# TODO: write a dataloader that takes as inputs coco and nifti files and outputs paired data for training
# where data is in the form of (image, annotation) where image is a 2D slice from the nifti file 
# and annotation is the corresponding activations from layer of ANN, given the coco image

from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch 
import numpy as np
import os

from diffusion_brain.datasets.paired_brain_ann_dataset import AnnActivationsDataset, PairedBrainAnnDataset
from diffusion_brain.utils.ann_activations_utils import ensure_activations_exist
from diffusion_brain.utils.fmri_behav_data_utils import get_train_test_subsets, get_train_test_indices, ensure_fmri_roi_exists

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
    train_dataset = PairedBrainAnnDataset(
        activations_path=train_activations_path,
        fmri_roi_path=fmri_roi_path,
        sample_indices=train_indices,
        is_2d=args.data.is_2d,
        act_index_map=train_act_index,
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
