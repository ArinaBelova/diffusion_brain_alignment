from sklearn.linear_model import RidgeCV
import torch 
from scipy.stats import pearsonr
import numpy as np
import rsatoolbox
import wandb
import os
#from fracridge import FracRidgeRegressorCV

from diffusion_brain.data_utils import get_dataloader
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.fmri_behav_data_utils import compute_rdm
from diffusion_brain.utils.visualise import pyplot_brain, get_r_across_images_2d_data, get_r_across_images_1d_data

def get_train_test_numpy_datasets(train_dataloader, test_dataloader, args):
    print(f"Length of the dataloaders: {len(train_dataloader)}, {len(test_dataloader)}", flush=True)
    # 2. Extract the single large batch
    train_fmri_dataset, train_activations_dataset = next(iter(train_dataloader))
    test_fmri_dataset, test_activations_dataset = next(iter(test_dataloader))

    return train_fmri_dataset.numpy(), test_fmri_dataset.numpy(),train_activations_dataset.numpy(), test_activations_dataset.numpy()

def preprocess_2d_dataset(dataset, args):
    # reshape the 2D fmri data to 1D (flatten the spatial dimensions)
    print("Dataset shape (before processing):", dataset.shape)  # Should be (num_images, H, W)
    locations_load_path = os.path.join(
            args.data.roi_defs_dir, f"roi_preselected_extended_2d_images_res_{args.data.grid_resolution_2d}", 
            f"{args.data.roi_file}",
            f"{args.data.subj}_{args.data.roi}.npz"
        )
    
    locations_roi = np.load(locations_load_path, allow_pickle=True)["locations"]  # [2, n_locations]
    y_coords = locations_roi[0]
    x_coords = locations_roi[1]
    
    dataset = dataset.squeeze(1)  # shape (num_images, H, W)
    dataset = dataset[:, y_coords, x_coords]  # shape (num_images, n_locations)
    print("Dataset shape after selecting ROI locations: ", dataset.shape, flush=True)
    
    return dataset

def train(train_activations_dataset, train_fmri_dataset, args):
    print("Training Ridge Regression with Cross-Validation...", flush=True)

    if args.data.is_2d:
        train_fmri_dataset = preprocess_2d_dataset(train_fmri_dataset, args)

    print("Stats of train activations dataset:", train_activations_dataset.mean(), train_activations_dataset.std(), flush=True)
    print("Stats of train fMRI dataset:", train_fmri_dataset.mean(), train_fmri_dataset.std(), flush=True)

    #alphas = np.logspace(-2, 6, 9)
    alphas = np.array([1e-3, 1e-2, 1e-1, 1])

    #################### DEBUG WITH PURE NOISE DATA ###################
    #train_activations_dataset = np.random.randn(*train_activations_dataset.shape)
    # train_fmri_dataset = np.random.randn(*train_fmri_dataset.shape)

    # permute rows (images) in activations matrix
    # np.random.shuffle(train_activations_dataset)
    ##############################################
    print("Train activations dataset shape: ", train_activations_dataset.shape, flush=True)
    print("Train fMRI dataset shape: ", train_fmri_dataset.shape, flush=True)

    clf = RidgeCV(alphas=alphas, scoring="r2").fit(train_activations_dataset, train_fmri_dataset)
    # , scoring="r2"
    print(f"Scoring used: {clf.scoring}")
    # clf = FracRidgeRegressorCV().fit(train_activations_dataset, train_fmri_dataset, frac_grid=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    # print("Learnt best frac parameter", clf.best_frac_, flush=True)
    print("Learn features shape: ", clf.coef_.shape, flush=True)
    print("Learned features stats: ", clf.coef_.mean(), clf.coef_.std(), flush=True)
    
    print("Training completed.", flush=True)
    print("Evaluating on training data...", flush=True)
    
    final_score = clf.score(train_activations_dataset, train_fmri_dataset)
    print("Evaluation completed.", flush=True)
    print(f"Best Alpha: {clf.alpha_}", flush=True)
    print(f"Score: {final_score}", flush=True)

    return clf

def validate_and_visualise(clf, true_activations_dataset, true_fmri_dataset, args, step="sklearn_fitting"):
    true_fmri_dataset_copy = true_fmri_dataset.copy()  # Make a copy to avoid modifying the original dataset
    if args.data.is_2d:
        # reshape the 2D fmri data to 1D (flatten the spatial dimensions)
        true_fmri_dataset = preprocess_2d_dataset(true_fmri_dataset, args)
        
    # validate and display the results
    print("Generating predicted fMRI data from activations...", flush=True)
    print("Shape of activations dataset:", true_activations_dataset.shape, flush=True)
    fmri_predicted = clf.predict(true_activations_dataset) 
    # print("Generated fmri samples shape: ", fmri_predicted.shape, flush=True)
    # print("Generated fmri samples type: ", type(fmri_predicted), flush=True)
    
    score_test = clf.score(true_activations_dataset, true_fmri_dataset)
    print("Score on the test data with TRUE FMRI: ", score_test, flush=True)

    ######## 1D Visualisation of predicted and true fMRI data for the test set ##########
    if args.data.is_2d:
        get_r_across_images_2d_data(args, fmri_predicted, true_fmri_dataset_copy, step_num=step)
    else:
        print("Visualising predicted fMRI data...", flush=True)
        # can only visualise one batch element
        pyplot_brain(fmri_predicted[0], args=args, savename=f"generated_sample_idx_{0}_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

        print("Visualising true fMRI data...", flush=True)
        pyplot_brain(true_fmri_dataset[0], args=args, savename=f"true_sample_idx_{0}_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

        print("Visualising the difference MSE between true and predicted fMRI data...", flush=True)
        print("Shape of true fMRI dataset:", true_fmri_dataset.shape, flush=True)
        difference = (true_fmri_dataset - fmri_predicted)**2
        pyplot_brain(difference.mean(axis=0), args=args, savename=f"roi_{args.data.roi}_difference_mse_step_{step}", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png')

        get_r_across_images_1d_data(args, fmri_predicted, true_fmri_dataset, step_num=step)
    ########## RDM visualisation on test dataset ##########
    print("Visualising RDM on test dataset: ", flush=True)
    rdms_true_test_fmri = compute_rdm(true_fmri_dataset, args, regime="test")
    rdms_predicted_test_fmri = compute_rdm(fmri_predicted, args, regime="test")

    wandb.log({"rdm_true_test_fmri": wandb.Image(rsatoolbox.vis.show_rdm(rdms_true_test_fmri)[0], caption="RDM True Test fMRI")})
    wandb.log({"rdm_predicted_test_fmri": wandb.Image(rsatoolbox.vis.show_rdm(rdms_predicted_test_fmri)[0], caption="RDM Predicted Test fMRI")})

def main():
    args = parse_args_and_setup_wandb()
    # just to get batch_size for the train dataloader correctly:
    train_dataloader, test_dataloader = get_dataloader(args)
    args.train.batch_size = len(train_dataloader.dataset) # set to the full training set size, not always 9485
    args.validation.batch_size = 515 # set to the full test set size
    # get correctly shaped dataloaders with the updated batch sizes:
    train_dataloader, test_dataloader = get_dataloader(args)
    print(f"Length of the TRAIN dataset: {args.train.batch_size}", flush=True)
    train_fmri_dataset, test_fmri_dataset, train_activations_dataset, test_activations_dataset = get_train_test_numpy_datasets(train_dataloader, test_dataloader, args)
    #################### TEST ###############################
    # train_fmri_dataset, _, train_activations_dataset, _ = get_train_test_numpy_datasets(train_dataloader, test_dataloader, args)
    # args.data.subj = "subj04"
    # train_dataloader, test_dataloader = get_dataloader(args)
    # args.train.batch_size = len(test_dataloader.dataset) # set to the full test set size
    # print("Length of TEST dataset: ", args.train.batch_size, flush=True)
    # train_dataloader, test_dataloader = get_dataloader(args)
    # _, test_fmri_dataset, _, test_activations_dataset = get_train_test_numpy_datasets(train_dataloader, test_dataloader, args)
    #########################################################
    # print("max of train fmri indexes:", train_fmri_dataset.indices.max(), flush=True)
    # print("max of test fmri indexes:", test_fmri_dataset.indices.max(), flush=True)
    # print("max of train activations indexes:", train_activations_dataset.indices.max(), flush=True)
    # print("max of test activations indexes:", test_activations_dataset.indices.max(), flush=True)
    # print()
    # print("min of train fmri indexes:", train_fmri_dataset.indices.min(), flush=True)
    # print("min of test fmri indexes:", test_fmri_dataset.indices.min(), flush=True)
    # print("min of train activations indexes:", train_activations_dataset.indices.min(), flush=True)
    # print("min of test activations indexes:", test_activations_dataset.indices.min(), flush=True)

    # print("Shape of train fmri dataset:", train_fmri_dataset.shape, flush=True)
    # print("Shape of train activations dataset:", train_activations_dataset.shape, flush=True)

    clf = train(train_activations_dataset, train_fmri_dataset, args)    

    #print("Shape of test fmri dataset:", test_fmri_dataset.shape, flush=True)
    #print("Shape of test activations dataset:", test_activations_dataset.shape, flush=True)
    validate_and_visualise(clf, test_activations_dataset, test_fmri_dataset, args)

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cpu") # sklearn works only on CPU
    main()   