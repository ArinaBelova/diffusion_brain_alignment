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

def get_ann_brain_dataloader(args):
    train_nsd_ids, test_nsd_ids, train_indices, test_indices = get_train_test_indices(args)
    
    # 2. Ensure fMRI ROI data exists (main process, thread-safe)
    fmri_roi_path = ensure_fmri_roi_exists(args)
    
    roi_indices_path = os.path.join(args.data.roi_defs_dir, "roi_indices", f"{args.data.roi_file}", f"{args.data.roi}.npy")
    roi_indices = np.load(roi_indices_path, allow_pickle=True)
    args.model.input_size = int(len(roi_indices))
    print(f"Setting model.input_size to ROI voxel count: {args.model.input_size}")
    
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

    # 4. Create dataset
    train_dataset = PairedBrainAnnDataset(
        activations_path=train_activations_path,
        fmri_roi_path=fmri_roi_path,
        sample_indices=train_indices,
    )

    test_dataset = PairedBrainAnnDataset(
        activations_path=test_activations_path,
        fmri_roi_path=fmri_roi_path,
        sample_indices=test_indices,
    )
    
    # 5. DataLoader
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