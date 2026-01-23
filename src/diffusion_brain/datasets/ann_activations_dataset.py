import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
import os

class H5Dataset(Dataset):
    # for the sake of our experimentation the data_name parameter is fixed to "imgBrick"
    def __init__(self, args, data_name="imgBrick"):
        self.file_path = os.path.join(args.data.images_data_path, "nsd_stimuli.hdf5")
        self.data_name = data_name
        #self.label_name = label_name
        self.transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((args.data.ann_input_size, args.data.ann_input_size), interpolation=transforms.InterpolationMode.BILINEAR),
                                            transforms.ToTensor()
                                        ]) # usually resizing to fit the image size expected by the ANN
        
        # Open the file once to get the length
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f[self.data_name])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # We open the file in 'r' mode inside __getitem__ 
        # to ensure it works with multi-process DataLoader
        with h5py.File(self.file_path, 'r') as f:
            data = f[self.data_name][idx]
            
            print("data type in ann activations dataset: ", data.dtype, flush=True)
            if self.transform:
                data = self.transform(data)

            # if self.label_name:
            #     label = torch.from_numpy(f[self.label_name][idx]).long()
            #    return data, label
            
            return data
        
class AnnActivationsDataset(H5Dataset):
    def __init__(self, args):
        super().__init__(args)
        self.layer_name = args.data.layer_name
        self.model = torch.hub.load('pytorch/vision', args.data.ann_model, pretrained=True, trust_repo=True) # model_name="resnet50"
        self.activations_extractor = create_feature_extractor(self.model, return_nodes={self.layer_name: 'feat'}) # layer_name: what we want to call it
        print(f"Initialized AnnActivationsDataset with layer: {self.layer_name}")

    def __getitem__(self, idx):
        # for now we assume that we don't need labels from our coco dataset
        print(f"INDEX IN ANN DATASET {idx}", flush=True)
        image = super().__getitem__(idx)
        
        # pass the image through the pretrained ANN to get activations
        activation = self._get_ann_activations(image)

        return activation
    
    # TODO maybe outsource choice of layer neame to the training script logic? 
    def _get_ann_activations(self, image):
        # Placeholder for actual ANN activation extraction logic
        # This should interface with the ANN model to get activations for the given layer
        self.model.eval()
        with torch.no_grad():
            activation = self.activations_extractor(image)

        return activation

