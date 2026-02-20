import torch
from torch.utils.data import Dataset
import os
from pathlib import Path


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


class PairedBrainAnnDataset(Dataset):
    """
    Pairs fMRI (condition) with ANN activations (target).
    Uses pre-computed ROI fMRI and aligned indices.
    """
    
    def __init__(self, activations_path, fmri_roi_path, sample_indices):
        """
        Args:
            activations_path: path to activations [n_split, *feat_dims]
            fmri_roi_path: path to ROI betas [n_total_samples, n_roi_voxels]
            sample_indices: indices into fmri for this split
        """
        self.activations = torch.load(Path(activations_path), map_location="cpu")
        
        # Load ROI betas and index
        fmri_all = torch.load(Path(fmri_roi_path), map_location="cpu")  # [n_total, n_voxels]
        self.fmri_data = fmri_all[sample_indices]  # [n_split, n_voxels]
        
        assert len(self.fmri_data) == len(self.activations), \
            f"Mismatch: fMRI={len(self.fmri_data)}, activations={len(self.activations)}"
        
        print(f"Dataset: {len(self)} samples, fMRI: {self.fmri_data.shape}, ANN: {self.activations.shape}")
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.fmri_data[idx], self.activations[idx]