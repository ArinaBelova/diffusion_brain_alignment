import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
import os
import wandb 
from PIL import Image
import numpy as np

class H5Dataset(Dataset):
    # for the sake of our experimentation the data_name parameter is fixed to "imgBrick"
    def __init__(self, args, data_name="imgBrick"):
        self.file_path = os.path.join(args.data.images_data_path, "nsd_stimuli.hdf5")
        self.data_name = data_name

        # Open the file once to get the length
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f[self.data_name])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # We open the file in 'r' mode inside __getitem__ 
        # to ensure it works with multi-process DataLoader
        with h5py.File(self.file_path, 'r') as f:
            data = f[self.data_name][idx - 1] # due to nature of extracting indices from our other functions
            data = Image.fromarray(data)
            
            return data
        
class AnnActivationsDataset(H5Dataset):
    def __init__(self, args):
        super().__init__(args)
        self.layer_name = args.data.layer_name
        self.model, self.transforms = get_ann_model(args)
        self.activations_extractor = create_feature_extractor(self.model, return_nodes={self.layer_name: 'feat'}) # layer_name: what we want to call it
        print(f"Initialized AnnActivationsDataset with layer: {self.layer_name}")

    def __getitem__(self, idx):
        # for now we assume that we don't need labels from our coco dataset
        image = super().__getitem__(idx)
        
        if self.transforms:
            image = self.transforms(image)

        ###################
        wandb.log({"image_for_activation_extraction": [wandb.Image(image * 255, caption=f"Index: {idx}")]})
        ###################
        # pass the image through the pretrained ANN to get activations
        # add batch dimension
        activation = self._get_ann_activations(image.unsqueeze(0)) # TODO: here we supply only 1 image, so we need to unsqueeze for batch dimension, but after we get activation we have batch_size, why?
        activation = activation['feat'].squeeze() # torch.Size([32, 2048])
        return activation
    
    # TODO maybe outsource choice of layer neame to the training script logic? 
    def _get_ann_activations(self, image):
        # Placeholder for actual ANN activation extraction logic
        # This should interface with the ANN model to get activations for the given layer
        with torch.no_grad():
            activation = self.activations_extractor(image)

        return activation

def get_ann_model(args):
    # 1. Load the weights object first using Hub
    # torch.hub.load('pytorch/vision', args.data.ann_model, pretrained=True, trust_repo=True) # model_name="resnet50"
    weights = torch.hub.load('pytorch/vision', 'get_weight', name=args.data.ann_model_weights)

    # 2. Get the transforms from that weight object
    transforms = weights.transforms()

    # 3. Load the model using those same weights
    model = torch.hub.load('pytorch/vision', args.data.ann_model, weights=weights)
    model.eval() 
    
    return model, transforms
