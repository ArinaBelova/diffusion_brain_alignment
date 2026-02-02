from sklearn.linear_model import RidgeCV
import torch 
from scipy.stats import pearsonr
import numpy as np

from diffusion_brain.data_utils import get_dataloader
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results
from diffusion_brain.utils.fmri_behav_data_utils import get_train_test_subsets
from diffusion_brain.datasets.fmri_betas_dataset import FmriDataset
from diffusion_brain.datasets.ann_activations_dataset import AnnActivationsDataset
from diffusion_brain.utils.visualise import pyplot_brain

def subset_to_numpy(subset):
    # 1. Create a loader for the entire subset at once
    print("Full dataset length:", len(subset), flush=True)
    loader = torch.utils.data.DataLoader(subset, batch_size=len(subset))
    print("Length of the dataloader:", len(loader), flush=True)
    # 2. Extract the single large batch
    inputs = next(iter(loader))
    
    # 3. Flatten if it's image/multidimensional data (e.g., from [N, 1, 28, 28] to [N, 784])
    if inputs.ndim > 2:
        inputs = inputs.view(inputs.size(0), -1)
    
    print("Shape of the input to the model: ", inputs.shape, flush=True)
    return inputs.numpy()

def train(train_activations_dataset, train_fmri_dataset):
    print("Training Ridge Regression with Cross-Validation...", flush=True)

    print("Shape of train activations dataset:", train_activations_dataset.shape, flush=True)
    print("Shape of train fMRI dataset:", train_fmri_dataset.shape, flush=True)

    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_activations_dataset, train_fmri_dataset)
    
    print("Training completed.", flush=True)
    print("Evaluating on training data...", flush=True)
    
    final_score = clf.score(train_activations_dataset, train_fmri_dataset)
    
    print("Evaluation completed.", flush=True)
    print(f"Best Alpha: {clf.alpha_}", flush=True)
    print(f"Score: {final_score}", flush=True)

    return clf

def validate_and_visualise(clf, activations_dataset, true_fmri_dataset, mean_roi, args, step="sklearn_fitting"):
    # validate and display the results
    print("Generating predicted fMRI data from activations...", flush=True)
    print("Shape of activations dataset:", activations_dataset.shape, flush=True)
    fmri_predicted = clf.predict(activations_dataset) 
    # print("Generated fmri samples shape: ", fmri_predicted.shape, flush=True)
    # print("Generated fmri samples type: ", type(fmri_predicted), flush=True)
    

    print("Visualising predicted fMRI data...", flush=True)
    # can only visualise one batch element
    pyplot_brain(fmri_predicted[0], args=args, savename=f"generated_sample_idx_{0}_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

    print("Visualising true fMRI data...", flush=True)
    pyplot_brain(true_fmri_dataset[0], args=args, savename=f"true_sample_idx_{0}_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

    print("Visualising the difference MSE between true and predicted fMRI data...", flush=True)
    print("Shape of true fMRI dataset:", true_fmri_dataset.shape, flush=True)
    difference = (true_fmri_dataset - fmri_predicted)**2
    pyplot_brain(difference.mean(axis=0), args=args, savename=f"roi_{args.data.roi}_difference_mse_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

    # todo correlation on test set
    # corr = pearsonr(fmri_predicted.ravel(), true_fmri_dataset.ravel())
    # pyplot_brain(corr[0], args=args, savename=f"roi_{args.data.roi}_correlation_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

def main():
    args = parse_args_and_setup_wandb()
    fmri_dataset = FmriDataset(args)

    mean_roi = fmri_dataset.data.mean(0)
    print(f"Mean of ROI {args.data.roi} across images ", mean_roi, flush=True)

    corrs = np.array([pearsonr(vox_idx, mean_roi) for vox_idx in fmri_dataset.data])[:,0]
    corr_pred_roi = corrs.mean(0)
    print("CORRELATIONS BETWEEN TRUE AND MEAN ROI ACTIVITY:", corr_pred_roi, flush=True)

    # activations_dataset = AnnActivationsDataset(args)
    # sum_, sum_sq, count = 0, 0, 0
    # for x in activations_dataset:      # or: for x in dataset
    #     sum_ += x.sum()
    #     sum_sq += (x ** 2).sum()
    #     count += x.numel()

    # mean = sum_ / count
    # std = torch.sqrt(sum_sq / count - mean ** 2)
    # print("MEAN OF ANN ACTIVATIONS DATASET ACROSS SAMPLES:", mean, flush=True)
    # print("STD OF ANN ACTIVATIONS DATASET ACROSS SAMPLES:", std, flush=True)

    train_fmri_dataset, test_fmri_dataset, train_activations_dataset, test_activations_dataset = get_train_test_subsets(fmri_dataset, activations_dataset, args)

    # print("max of train fmri indexes:", train_fmri_dataset.indices.max(), flush=True)
    # print("max of test fmri indexes:", test_fmri_dataset.indices.max(), flush=True)
    # print("max of train activations indexes:", train_activations_dataset.indices.max(), flush=True)
    # print("max of test activations indexes:", test_activations_dataset.indices.max(), flush=True)
    # print()
    # print("min of train fmri indexes:", train_fmri_dataset.indices.min(), flush=True)
    # print("min of test fmri indexes:", test_fmri_dataset.indices.min(), flush=True)
    # print("min of train activations indexes:", train_activations_dataset.indices.min(), flush=True)
    # print("min of test activations indexes:", test_activations_dataset.indices.min(), flush=True)

    train_fmri_dataset, train_activations_dataset = subset_to_numpy(train_fmri_dataset), subset_to_numpy(train_activations_dataset)
    test_fmri_dataset, test_activations_dataset = subset_to_numpy(test_fmri_dataset), subset_to_numpy(test_activations_dataset)

    # print("Shape of train fmri dataset:", train_fmri_dataset.shape, flush=True)
    # print("Shape of train activations dataset:", train_activations_dataset.shape, flush=True)

    clf = train(train_activations_dataset, train_fmri_dataset)    

    #print("Shape of test fmri dataset:", test_fmri_dataset.shape, flush=True)
    #print("Shape of test activations dataset:", test_activations_dataset.shape, flush=True)
    validate_and_visualise(clf, test_activations_dataset, test_fmri_dataset, mean_roi, args)

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cpu") # sklearn works only on CPU
    main()   