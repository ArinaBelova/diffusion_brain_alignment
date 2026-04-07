import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np

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
        is_2d=False,
        act_mean=None,
        act_std=None,
        fmri_scale=None,
        act_index_map=None,
        fmri_norm_mode="active_std",
    ):
        """
        Args:
            activations_path: path to activations [n_unique, *feat_dims]
            fmri_roi_path: path to ROI betas [n_total_samples, n_roi_voxels]
            sample_indices: indices into fmri for this split
            is_2d: if True, load pre-processed 2D images
            act_mean: pre-computed ANN activation mean (from train set). If None, computed from this split.
            act_std: pre-computed ANN activation std (from train set). If None, computed from this split.
            fmri_scale: pre-computed fMRI normalisation scale (from train set). If None, computed from this split.
            act_index_map: optional array of shape (n_fmri_rows,) mapping each fMRI
                row to an index in the activations tensor.  Used for the unaveraged
                dataset where multiple fMRI repetitions share one ANN activation.
                If None, a 1:1 mapping is assumed (averaged dataset).
            fmri_norm_mode: "active_std" divides by std of non-zero voxels (active
                voxels get std≈1, full 2D image std≈0.3 due to zero-padding).
                "global_std" divides by std of the full image including zeros (full
                image std≈1, matching the VP-SDE noise schedule assumption).
        """
        self.activations = torch.load(Path(activations_path), map_location="cpu")
        self.act_index_map = act_index_map  # None → identity mapping
        self.is_2d = is_2d
        self.fmri_norm_mode = fmri_norm_mode
        self.act_mean = act_mean if act_mean is not None else self.activations.mean(dim=0)
        self.act_std = act_std if act_std is not None else self.activations.std(dim=0)

        if self.is_2d:
            fmri_data = np.load(fmri_roi_path, allow_pickle=True)["data"]
            self.fmri_data = fmri_data[sample_indices]  # [n_split, 1, H, W]
        else:
            fmri_all = torch.load(Path(fmri_roi_path), map_location="cpu")
            self.fmri_data = fmri_all[sample_indices] # [n_split, n_voxels]

        # Normalisation scale.
        # "active_std": std over active (non-zero) voxels → active voxels std≈1,
        #   but full 2D image std≈0.3 due to zero-padding sparsity.
        # "global_std": std over the full image including zeros → full image std≈1,
        #   matching the VP-SDE assumption that data has unit variance.
        if fmri_scale is not None:
            self.fmri_scale = fmri_scale
        else:
            if fmri_norm_mode == "global_std":
                if isinstance(self.fmri_data, np.ndarray):
                    self.fmri_scale = float(np.std(self.fmri_data)) if self.fmri_data.size > 0 else 1.0
                else:
                    self.fmri_scale = float(self.fmri_data.std()) if self.fmri_data.numel() > 0 else 1.0
            else:  # "active_std" (default)
                if self.is_2d:
                    active_vals = self.fmri_data[self.fmri_data != 0]
                else:
                    active_vals = self.fmri_data[self.fmri_data != 0] if isinstance(self.fmri_data, np.ndarray) else self.fmri_data[self.fmri_data != 0]
                if isinstance(active_vals, torch.Tensor):
                    self.fmri_scale = float(active_vals.std()) if len(active_vals) > 0 else 1.0
                else:
                    self.fmri_scale = float(np.std(active_vals)) if len(active_vals) > 0 else 1.0
        print(f"fMRI normalisation mode: {fmri_norm_mode}, scale: {self.fmri_scale:.4f}", flush=True)

        # Store normalised fMRI range for thresholding during generation
        normed = self.fmri_data / self.fmri_scale
        self.fmri_min = float(normed.min())
        self.fmri_max = float(normed.max())

        print(f"Statistics of fMRI data: max={self.fmri_data.max():.4f}, min={self.fmri_data.min():.4f}, mean={self.fmri_data.mean():.4f}, std={self.fmri_data.std():.4f}", flush=True)
        print(f"Normalised fMRI range for thresholding: [{self.fmri_min:.4f}, {self.fmri_max:.4f}]", flush=True)
        print(f"Statistics of normalised fMRI data: max={normed.max():.4f}, min={normed.min():.4f}, mean={normed.mean():.4f}, std={normed.std():.4f}", flush=True)
        print(f"Statistics of ANN activations: max={self.activations.max():.4f}, min={self.activations.min():.4f}, mean={self.activations.mean():.4f}, std={self.activations.std():.4f}", flush=True)

        if self.act_index_map is not None:
            assert len(self.fmri_data) == len(self.act_index_map), \
                f"Mismatch: fMRI={len(self.fmri_data)}, act_index_map={len(self.act_index_map)}"
            print(f"Dataset (unaveraged): {len(self)} fMRI samples, {len(self.activations)} unique activations, ANN: {self.activations.shape}")
        else:
            assert len(self.fmri_data) == len(self.activations), \
                f"Mismatch: fMRI={len(self.fmri_data)}, activations={len(self.activations)}"
            print(f"Dataset: {len(self)} samples, fMRI: {self.fmri_data.shape}, ANN: {self.activations.shape}")

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri = self.fmri_data[idx] / self.fmri_scale
        act_idx = self.act_index_map[idx] if self.act_index_map is not None else idx
        act = self.activations[act_idx]
        act = (act - self.act_mean) / (self.act_std + 1e-6)  # z-score per feature
        return fmri, act