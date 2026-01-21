import torch

import src.diffusion_brain.data_utils.datasets.coco_dataset as coco_dataset
from torchvision.models.feature_extraction import create_feature_extractor

class AnnActivationsDataset(coco_dataset.H5Dataset):
    def __init__(self, file_path, model_name, layer_name):
        super().__init__(file_path, data_name=layer_name)
        self.layer_name = layer_name
        self.model = torch.hub.load('pytorch/vision', model_name, pretrained=True, trust_repo=True) # model_name="resnet50"
        self.activations_extractor = create_feature_extractor(self.model, return_nodes={self.layer_name: 'feat'}) # layer_name: what we want to call it
        print(f"Initialized AnnActivationsDataset with layer: {self.layer_name}")

    def __getitem__(self, idx):
        # for now we assume that we don't need labels from our coco dataset
        image = super().__getitem__(idx)
        
        # pass the image through the pretrained ANN to get activations
        activation = self._get_ann_activations(image)

        return activation
    
    # TODO@ maybe outsource choice of layer neame to the training script logic? 
    def _get_ann_activations(self, image):
        # Placeholder for actual ANN activation extraction logic
        # This should interface with the ANN model to get activations for the given layer
        self.model.eval()
        with torch.no_grad():
            activation = self.activations_extractor(image)

        return activation
