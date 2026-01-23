import numpy as np
import os
import nibabel as nb
import torch
from torch.utils.data import Dataset

def get_roi_indices(args):
    # 1. Load the ROI mask (using your existing logic)
    print("Loading ROI masks...")
    maskdata, _ = get_rois(args.data.roi_defs_dir, args.data.roi_file, args.data.roi)
    
    roi_indices = np.where(maskdata == args.data.roi)[0]
    print(f"Extracted {len(roi_indices)} voxels for ROI: {args.data.roi}")
    return roi_indices

def preprocess_nsd_roi(args):
    # 1. Get ROI indices 
    roi_indices = get_roi_indices(args)

    # 2. Load the large beta file
    data_name = f"{args.data.subj}_" + f"{args.data.fmri_data_name}"
    full_path = os.path.join(args.data.fmri_data_root, data_name)
    print(f"Loading full dataset from {full_path}...")
    
    # We load with mmap first to avoid crashing RAM if the file is huge
    full_data = np.load(full_path, mmap_mode='r') 
    
    # 3. Slice
    # This pulls only the necessary data into RAM and reorganizes it
    # to be [n_voxels_in_roi, n_samples]
    processed_data = full_data[roi_indices, :]
    
    # 4. Save as a new file
    output_path = os.path.join(args.data.roi_defs_dir, f"roi_preselected", f"{args.data.roi_file}", data_name, f"{args.data.subj}_{args.data.roi}.npy")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    np.save(output_path, processed_data)
    print(f"Saved optimized dataset to: {output_path}")
    
    return output_path

def get_rois(roi_defs_dir, roi_file, roi):
        roi_names_file = os.path.join(roi_defs_dir, f"{roi_file}.mgz.ctab")
        try:
            with open(roi_names_file) as f:
                # get ROI names automatically. If you don't have the .ctab file
                # you can also enter them by hand. 0 is always "Unknown")
                roi_id2name = {int(x[0]): x[2:-1] for x in f}
        except ValueError:
            print(
                f"roi_names_file not found. Requested {roi_names_file}. Using {roi} as single ROI name."
            )
            roi_id2name = {0: "Unknown"}
            roi_id2name[1] = roi

        # load the roi masks
        try:
            lh_file = os.path.join(roi_defs_dir, f"lh.{roi_file}.mgz")
            rh_file = os.path.join(roi_defs_dir, f"rh.{roi_file}.mgz")
            maskdata_lh = nb.load(lh_file).get_fdata().squeeze()
            maskdata_rh = nb.load(rh_file).get_fdata().squeeze()
        except ValueError:
            lh_file = os.path.join(roi_defs_dir, f"lh.{roi_file}.npy")
            rh_file = os.path.join(roi_defs_dir, f"rh.{roi_file}.npy")
            maskdata_lh = np.load(lh_file, allow_pickle=True)
            maskdata_rh = np.load(rh_file, allow_pickle=True)

        maskdata = np.hstack((maskdata_lh, maskdata_rh))

        return maskdata, roi_id2name        

class FmriDataset(Dataset):
    def __init__(self, args):
        # We don't even need the ROI logic here anymore!
        # Just point to the file created by the script above.
        processed_file_path = os.path.join(args.data.roi_defs_dir, f"roi_preselected", f"{args.data.roi_file}", f"{args.data.subj}_{args.data.roi}.npy")
        if not os.path.exists(processed_file_path):
            print(f"Processed file not found at {processed_file_path}. Running preprocessing...")
            processed_file_path = preprocess_nsd_roi(args)
        
        self.data = np.load(processed_file_path, mmap_mode='r').T # I want dataset here to be [n_img, n_voxels], so we can sample by image 
        self.roi_indices = get_roi_indices(args)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # This read is now lightning fast and perfectly 
        # compatible with multi-processing.
        return torch.from_numpy(self.data[idx].copy()).float()