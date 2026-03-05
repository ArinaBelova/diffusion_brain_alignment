import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
from diffusion_brain.utils.fmri_behav_data_utils import precompute_roi_indices, signal_to_2d
import numpy as np
import pickle

class AnnActivationsDataset(Dataset):
    """Multi-worker safe dataset for pre-computed activations."""
    
    def __init__(self, activations_path):
        if not os.path.isfile(activations_path):
            raise FileNotFoundError(
                f"Activations not found at {activations_path}. "
                f"Call ensure_activations_exist() before creating DataLoader."
            )
        
        # Load as tensor [n_samples, *feature_dims]
        self.activations = torch.load(activations_path, map_location="cpu")
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]


# class PairedBrainAnnDataset(Dataset):
#     """
#     Pairs fMRI (condition) with ANN activations (target).
#     Uses pre-computed ROI fMRI and aligned indices.
#     """
    
#     def __init__(self, activations_path, fmri_roi_path, sample_indices):
#         """
#         Args:
#             activations_path: path to activations [n_split, *feat_dims]
#             fmri_roi_path: path to ROI betas [n_total_samples, n_roi_voxels]
#             sample_indices: indices into fmri for this split
#         """
#         self.activations = torch.load(Path(activations_path), map_location="cpu")
        
#         # Load ROI betas and index
#         fmri_all = torch.load(Path(fmri_roi_path), map_location="cpu")  # [n_total, n_voxels]
#         self.fmri_data = fmri_all[sample_indices]  # [n_split, n_voxels]
        
#         assert len(self.fmri_data) == len(self.activations), \
#             f"Mismatch: fMRI={len(self.fmri_data)}, activations={len(self.activations)}"
        
#         print(f"Dataset: {len(self)} samples, fMRI: {self.fmri_data.shape}, ANN: {self.activations.shape}")
    
#     def __len__(self):
#         return len(self.activations)
    
#     def __getitem__(self, idx):
#         return self.fmri_data[idx], self.activations[idx]

class PairedBrainAnnDataset(Dataset):
    """
    Pairs fMRI (condition) with ANN activations (target).
    Loads either 1D or pre-processed 2D fMRI data.
    """
    
    def __init__(
        self,
        activations_path,
        fmri_roi_path,
        sample_indices,
        fmri_roi_2d_path=None,
        is_2d=False,
    ):
        """
        Args:
            activations_path: path to activations [n_split, *feat_dims]
            fmri_roi_path: path to ROI betas [n_total_samples, n_roi_voxels]
            sample_indices: indices into fmri for this split
            fmri_roi_2d_path: path to pre-processed 2D images [n_total_samples, H, W]
            is_2d: if True, load pre-processed 2D images (requires fmri_roi_2d_path)
        """
        self.activations = torch.load(Path(activations_path), map_location="cpu")
        self.is_2d = is_2d
        
        # Load fMRI data (1D or pre-processed 2D)
        if is_2d:
            if not fmri_roi_2d_path or not os.path.isfile(fmri_roi_2d_path):
                raise FileNotFoundError(
                    f"2D fMRI data not found at {fmri_roi_2d_path}. "
                    f"Please run preprocessing with is_2d=true first."
                )
            print(f"Loading pre-processed 2D fMRI images from {fmri_roi_2d_path}")
            # TODO@ maybe also load here the extended roi_info information so I can use it to transform the 2D images back to 1D for the evaluation stage
            fmri_2d_all = torch.load(Path(fmri_roi_2d_path), map_location="cpu")
            self.fmri_data = fmri_2d_all[sample_indices]
        else:
            fmri_all = torch.load(Path(fmri_roi_path), map_location="cpu")
            self.fmri_data = fmri_all[sample_indices]
        
        assert len(self.fmri_data) == len(self.activations), \
            f"Mismatch: fMRI={len(self.fmri_data)}, activations={len(self.activations)}"
        
        print(f"Dataset: {len(self)} samples, fMRI: {self.fmri_data.shape}, ANN: {self.activations.shape}")
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.fmri_data[idx], self.activations[idx]