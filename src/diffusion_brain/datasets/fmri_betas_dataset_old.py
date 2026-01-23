import nibabel as nb
from torch.utils.data import Dataset
import os
import numpy as np
import torch 

class FmriDataset(Dataset):
    def __init__(self, args):
        """
        file_path: path to pre-processed directory
        """
        # for now I'll use the overall file and don't touch the separation into 515 and NOT515 files
        # separation will happen in the dataloader logic later
        self.file_path = args.data.fmri_data_root + args.data.subj + "_betas_average_fsaverage.npy"
        self.roi = args.data.roi  
        if self.roi:
            self.roi_defs_dir = args.data.roi_defs_dir
            self.roi_file = args.data.roi_file
            self.maskdata, self.roi_id2name = self._get_rois()
            # Store only the indices to avoid passing the whole mask array
            self.roi_indices = np.where(self.maskdata == self.roi)[0]
        else:
            self.roi_indices = None    

        # Use a lazy-loading strategy for the mmap
        self.mmap_data = None    

    def __len__(self):
        return 10000

    # TODO: it makes sense to save the ROI files as separate .npy and then to load them here directly
    def __getitem__(self, idx):
        # Initialize mmap once per worker process
        if self.mmap_data is None:
            self.mmap_data = np.load(self.file_path, mmap_mode='r')

        # 2. Slice the mmap using pre-calculated indices
        if self.roi_indices is not None:
            # Slicing the specific sample AND specific ROI voxels
            # It's faster to index the specific sample first [idx] 
            # then filter by voxels [self.roi_indices]
            sample = self.mmap_data[self.roi_indices, idx]
        else:
            sample = self.mmap_data[:, idx]

        # Use .copy() to bring the memory-mapped slice into a standard numpy array
        # so PyTorch can convert it to a tensor safely.
        return torch.from_numpy(sample.copy())  #.float()  
    
    # get_roi function returns overall mask (flattened array of voxels shape) with all the ROIs labelled as defined in *.mgz.ctab file 
    def _get_rois(self):
        roi_names_file = os.path.join(self.roi_defs_dir, f"{self.roi_file}.mgz.ctab")
        try:
            with open(roi_names_file) as f:
                # get ROI names automatically. If you don't have the .ctab file
                # you can also enter them by hand. 0 is always "Unknown")
                roi_id2name = {int(x[0]): x[2:-1] for x in f}
        except ValueError:
            print(
                f"roi_names_file not found. Requested {roi_names_file}. Using {self.roi} as single ROI name."
            )
            roi_id2name = {0: "Unknown"}
            roi_id2name[1] = self.roi

        # load the roi masks
        try:
            lh_file = os.path.join(self.roi_defs_dir, f"lh.{self.roi_file}.mgz")
            rh_file = os.path.join(self.roi_defs_dir, f"rh.{self.roi_file}.mgz")
            maskdata_lh = nb.load(lh_file).get_fdata().squeeze()
            maskdata_rh = nb.load(rh_file).get_fdata().squeeze()
        except ValueError:
            lh_file = os.path.join(self.roi_defs_dir, f"lh.{self.roi_file}.npy")
            rh_file = os.path.join(self.roi_defs_dir, f"rh.{self.roi_file}.npy")
            maskdata_lh = np.load(lh_file, allow_pickle=True)
            maskdata_rh = np.load(rh_file, allow_pickle=True)

        maskdata = np.hstack((maskdata_lh, maskdata_rh))

        return maskdata, roi_id2name    