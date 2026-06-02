import hashlib

import torch
import h5py
from torch.utils.data import Dataset
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models

import os
import numpy as np
from PIL import Image
from filelock import FileLock
from pathlib import Path

from diffusion_brain.utils.fmri_behav_data_utils import _user_scoped_lock_path


def _nsd_ids_hash(nsd_ids):
    """Return a short hex hash of the NSD ID array for cache-safe filenames."""
    raw = np.array(sorted(nsd_ids), dtype=np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()[:12]

def load_model(model_name: str, weights_name: str = "DEFAULT"):
    # Get model constructor and weights class
    model_fn = getattr(models, model_name)
    weights_enum = models.get_model_weights(model_name)
    
    # Get specific weights
    weights = getattr(weights_enum, weights_name)
    
    # Load model
    model = model_fn(weights=weights)
    model.eval()
    
    return model, weights.transforms()

def precompute_activations(indices_to_extract, args, data_name="imgBrick", save_path=None):
    """
    Extract activations - MUST be called from main process before DataLoader.
    Returns tensor of shape [n_samples, *feature_dims]
    """
    device = torch.device("cpu")
    
    # Load model
    weights_name = args.data.ann_model_weights
    model_name = args.data.ann_model
    print(f"In precompute activations the requested weights are {weights_name}")
    # weights = torch.hub.load('pytorch/vision', 'get_weight', name=weights_name)
    # transforms = weights.transforms()
    # model = torch.hub.load('pytorch/vision', args.data.ann_model, weights=weights)
    
    model, transforms = load_model(model_name, weights_name)
    print(f"Transforms of the model {model_name} are {transforms}")
    is_vit_model = model_name.startswith("vit_")

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
                feat = extractor(img_tensor)['feat']
                if is_vit_model and feat.ndim == 3:
                    feat = feat[:, 0, :].squeeze(0).cpu()
                else:
                    feat = feat.squeeze().cpu()
                activations.append(feat)
    
    # Stack into tensor [n_samples, *feature_dims]
    activations = torch.stack(activations, dim=0)
    
    # Save
    if save_path is None:
        h = _nsd_ids_hash(indices_to_extract)
        save_path = os.path.join(
            args.data.ann_activations_data_path,
            model_name,
            f"activations_weights_{weights_name}_layer_{args.data.layer_name}_{len(indices)}_samples_{h}.pt"
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activations, save_path)
    print(f"Saved activations to {save_path}, shape: {activations.shape}")
    
    return activations, save_path


def ensure_activations_exist(activations_path, indices_to_extract, args):
    """Thread-safe check and extraction.

    Fast path: if the activations cache already exists, return without
    touching any lock. This avoids PermissionError on shared caches where a
    different user owns the `.lock` sentinel. First-time creation uses a
    per-user lock under `$TMPDIR/diffusion_brain_locks_<user>/`.
    """
    if os.path.isfile(activations_path):
        print(f"Activations found at {activations_path}")
        return

    lock_path = _user_scoped_lock_path(activations_path)

    with FileLock(lock_path):
        if not os.path.isfile(activations_path):
            print(f"Activations not found at {activations_path}. Extracting...")
            precompute_activations(indices_to_extract, args, save_path=activations_path)
        else:
            print(f"Activations found at {activations_path}")
