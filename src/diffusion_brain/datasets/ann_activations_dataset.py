import torch
import h5py
from torch.utils.data import Dataset
from torchvision.models.feature_extraction import create_feature_extractor
import os
import numpy as np
from PIL import Image
from filelock import FileLock
from pathlib import Path

def precompute_activations(indices_to_extract, args, data_name="imgBrick"):
    """
    Extract activations - MUST be called from main process before DataLoader.
    Returns tensor of shape [n_samples, *feature_dims]
    """
    device = torch.device("cpu")
    
    # Load model
    weights_name = args.data.ann_model_weights
    print(f"In precompute activations the requested weights are {weights_name}")
    weights = torch.hub.load('pytorch/vision', 'get_weight', name=weights_name)
    transforms = weights.transforms()
    print(f"Transforms of the model {args.data.ann_model} are {transforms}")
    model = torch.hub.load('pytorch/vision', args.data.ann_model, weights=weights)
    
    extractor = create_feature_extractor(
        model,
        return_nodes={args.data.layer_name: 'feat'}
    ).to(device).eval()
    
    # Open H5 file
    file_path = os.path.join(args.data.images_data_path, "nsd_stimuli.hdf5")
    indices = np.array(indices_to_extract) # - 1  
    
    activations = []  # List instead of dict
    
    with h5py.File(file_path, 'r') as f:
        dataset = f[data_name]
        
        with torch.no_grad():
            for nsd_idx in indices:
                img = Image.fromarray(dataset[nsd_idx - 1]) # 1-indexed to 0-indexed
                img_tensor = transforms(img).unsqueeze(0).to(device)
                feat = extractor(img_tensor)['feat'].squeeze().cpu()
                activations.append(feat)
    
    # Stack into tensor [n_samples, *feature_dims]
    activations = torch.stack(activations, dim=0)
    
    # Save
    save_path = os.path.join(
        args.data.ann_activations_data_path, 
        args.data.ann_model,
        f"activations_weights_{weights_name}_layer_{args.data.layer_name}_{len(indices)}_samples.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activations, save_path)
    print(f"Saved activations to {save_path}, shape: {activations.shape}")
    
    return activations, save_path


def ensure_activations_exist(activations_path, indices_to_extract, args):
    """Thread-safe check and extraction."""
    lock_path = activations_path + ".lock"
    
    with FileLock(lock_path):
        if not os.path.isfile(activations_path):
            print(f"Activations not found at {activations_path}. Extracting...")
            precompute_activations(indices_to_extract, args)
        else:
            print(f"Activations found at {activations_path}")


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