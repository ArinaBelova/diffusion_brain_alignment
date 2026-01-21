import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, file_path, data_name, label_name=None):
        self.file_path = file_path
        self.data_name = data_name
        self.label_name = label_name
        
        # Open the file once to get the length
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f[self.data_name])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # We open the file in 'r' mode inside __getitem__ 
        # to ensure it works with multi-process DataLoader
        with h5py.File(self.file_path, 'r') as f:
            data = torch.from_numpy(f[self.data_name][idx]).float()
            
            if self.label_name:
                label = torch.from_numpy(f[self.label_name][idx]).long()
                return data, label
            
            return data       