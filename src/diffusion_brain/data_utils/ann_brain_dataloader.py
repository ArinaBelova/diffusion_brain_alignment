# TODO: write a dataloader that takes as inputs coco and nifti files and outputs paired data for training
# where data is in the form of (image, annotation) where image is a 2D slice from the nifti file 
# and annotation is the corresponding activations from layer of ANN, given the coco image

from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch 
import numpy as np
import os

from diffusion_brain.datasets.fmri_betas_dataset import FmriDataset
from diffusion_brain.datasets.ann_activations_dataset import AnnActivationsDataset
from diffusion_brain.utils.fmri_behav_data_utils import get_train_test_subsets

class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return len(self.datasets[0])

def get_ann_brain_dataloder(args):
    fmri_dataset = FmriDataset(args)
    activations_dataset = AnnActivationsDataset(args)

    train_fmri_dataset, test_fmri_dataset, train_activations_dataset, test_activations_dataset = get_train_test_subsets(fmri_dataset, activations_dataset, args)

    train_dataset = ZipDataset(train_fmri_dataset, train_activations_dataset) # instead of TensorDataset, which throws an error
    test_dataset = ZipDataset(test_fmri_dataset, test_activations_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.validation.batch_size, shuffle=False, num_workers=args.validation.num_workers, drop_last=False)

    return train_dataloader, test_dataloader