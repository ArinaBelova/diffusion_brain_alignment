from sklearn.linear_model import RidgeCV
import torch 

from diffusion_brain.data_utils import get_dataloader
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results
from diffusion_brain.utils.fmri_behav_data_utils import get_train_test_subsets
from diffusion_brain.datasets.fmri_betas_dataset import FmriDataset
from diffusion_brain.datasets.ann_activations_dataset import AnnActivationsDataset

def subset_to_numpy(subset):
    # 1. Create a loader for the entire subset at once
    loader = torch.utils.data.DataLoader(subset, batch_size=len(subset))
    
    # 2. Extract the single large batch
    inputs, targets = next(iter(loader))
    
    # 3. Flatten if it's image/multidimensional data (e.g., from [N, 1, 28, 28] to [N, 784])
    if inputs.ndim > 2:
        inputs = inputs.view(inputs.size(0), -1)
        
    return inputs.numpy(), targets.numpy()

def train(args):
    fmri_dataset = FmriDataset(args)
    activations_dataset = AnnActivationsDataset(args)
    train_fmri_dataset, test_fmri_dataset, train_activations_dataset, test_activations_dataset = get_train_test_subsets(fmri_dataset, activations_dataset, args)

    train_fmri_dataset, train_activations_dataset = subset_to_numpy(train_fmri_dataset), subset_to_numpy(train_activations_dataset)

    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_activations_dataset, train_fmri_dataset)
    final_score = clf.score(train_activations_dataset, train_fmri_dataset)
    print(f"Best Alpha: {clf.alpha_}")
    print(f"Score: {final_score}")

def main():
    args = parse_args_and_setup_wandb()
    train(args)    

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cpu") # sklearn works only on CPU
    main()   