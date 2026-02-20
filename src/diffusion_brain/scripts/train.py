from types import SimpleNamespace
import torch 
import torchvision
import wandb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import itertools
import os
import torch.nn.functional as F
from scipy.stats import pearsonr

from diffusion_brain.utils.grad_updaters import set_loss_function, set_optimiser, set_learning_rate_scheduler 
from diffusion_brain.models import set_model
from diffusion_brain.models.autoencoder import LinearAutoencoder
from diffusion_brain.data_utils import get_dataloader
import diffusion_brain.utils.diffusivity as diffusivity
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results, pyplot_brain
from diffusion_brain.utils.fmri_behav_data_utils import ensure_fmri_roi_exists

def pad_1d_to_factor(x, factor):
    if factor <= 1:
        return x, 0
    length = x.shape[-1]
    pad_len = (factor - (length % factor)) % factor
    if pad_len == 0:
        return x, 0
    return F.pad(x, (0, pad_len)), pad_len


def train_linear_autoencoder(args, train_dataloader, device):
    use_autoencoder = getattr(args.model, "use_autoencoder", args.model.name == "gfdm-unet-1d-cond")
    if not use_autoencoder:
        return None

    latent_dim = getattr(args.model, "ae_latent_dim", 512)
    ae_epochs = getattr(args.model, "ae_epochs", 10)
    ae_lr = getattr(args.model, "ae_lr", 1e-3)
    ae_save_path = getattr(args.model, "ae_save_path", None)
    ae_name = getattr(args.model, "ae_name", "ae_linear.pt")

    ae_save_path = os.path.join(ae_save_path, args.data.roi_file, ae_name + f"roi_{str(args.data.roi)}" + ".pth") if ae_save_path is not None else None

    input_dim = args.model.input_size
    autoencoder = LinearAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

    if ae_save_path is not None and os.path.isfile(ae_save_path):
        autoencoder.load_state_dict(torch.load(ae_save_path, map_location=device))
        autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad = False
        print(f"Loaded autoencoder from {ae_save_path}")
        args.model.ae_latent_dim = latent_dim
        return autoencoder

    print(f"Training linear autoencoder: input_dim={input_dim}, latent_dim={latent_dim}, epochs={ae_epochs}")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=ae_lr)
    loss_fn = torch.nn.MSELoss()

    autoencoder.train()
    for epoch in range(ae_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for data, _ in train_dataloader:
            x = data.float().to(device)
            if x.ndim == 3:
                x = x.squeeze(1)
            optimizer.zero_grad()
            x_hat, _ = autoencoder(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"AE epoch {epoch + 1}/{ae_epochs} loss: {avg_loss:.6f}")
        wandb.log({"autoencoder/loss": avg_loss, "autoencoder/epoch": epoch + 1})

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    print(f"Autoencoder training completed. Latent dim: {latent_dim}")
    
    # Log a sample reconstruction to wandb
    autoencoder.eval()
    with torch.no_grad():
        sample_data, _ = next(iter(train_dataloader))
        sample_data = sample_data[0].float().to(device) # take the first element of the batch
        if sample_data.ndim == 3:
            sample_data = sample_data.squeeze(1)
        
        latent = autoencoder.encoder(sample_data)
        reconstructed = autoencoder.decoder(latent)
        
        pyplot_brain(sample_data.cpu().squeeze().numpy(), args=args, savename="original_sample", figpath=f"{args.validation.output_folder}/{args.jobid}/autoencoder", save_type='png')
        pyplot_brain(reconstructed.cpu().squeeze().numpy(), args=args, savename="reconstructed_sample", figpath=f"{args.validation.output_folder}/{args.jobid}/autoencoder", save_type='png')

    if ae_save_path is not None:
        os.makedirs(os.path.dirname(ae_save_path), exist_ok=True)
        torch.save(autoencoder.state_dict(), ae_save_path)
        print(f"Saved autoencoder to {ae_save_path}")

    args.model.ae_latent_dim = latent_dim
    return autoencoder


def one_step_score_estimation(x, t, noise, label, score_fn, loss_function, diffusion_process, args, orig_len=None):
    # TODO: check this! here I simply need to estimate p_{0t}(x(t)|x(0)) mean and variance and use them to compute the true score    
    true_score = -noise
    mu, std = diffusion_process.brown_moments(x, t)
    x_t = mu + std * noise

    mask = torch.bernoulli(torch.full((len(label),), args.model.dropout_prob)).to(label.device) 

    if args.data.data_name == "ann-brain":
        # for ann-brain data we have a continuous label vector representation instead of discrete num_classes
        masked_labels = label * (1 - mask[:, None])
        #print("Masked labels shape in score estimation: ", masked_labels.shape, flush=True)
        encoder_hidden_states = masked_labels.unsqueeze(1).float()
        class_labels_arg = None
    else:    
        masked_labels = label * (1 - mask) + (args.model.num_classes * mask) # don't use -1 as the empty label as nn.Embedding will throw error
        masked_labels = masked_labels.long()    

    if args.model.name in ["unet-diffusers", "unet-diffusers-1d"]:
        # class_embeddings = score_fn.class_embedding(masked_labels)  # shape [bs, cross_attn_dim]
        # class_embeddings = class_embeddings.unsqueeze(1)  # shape [bs, 1, cross_attn_dim]
        # print(f"encoder_hidden_states shape: {class_embeddings.shape}")
        # print(x_t.shape, t.shape, masked_labels.shape, encoder_hidden_states.shape)


        #######################
        t = t * 999
        #######################

        #encoder_hidden_states = torch.zeros(x_t.shape[0], 1, args.model.cross_attention_dim, device=x_t.device)

        predicted_score = score_fn(x_t, t, encoder_hidden_states = encoder_hidden_states, class_labels=class_labels_arg).sample    # class_labels=masked_labels  and zeros encoder_hidden_states for MNIST case
    elif args.model.name == "gfdm-unet-1d-cond":
        predicted_score = score_fn(x_t, t, masked_labels.float())
    else:
        predicted_score = score_fn(x_t, t, masked_labels)

    if orig_len is not None:
        predicted_score = predicted_score[..., :orig_len]
        true_score = true_score[..., :orig_len]

    loss = loss_function(predicted_score, true_score)
    
    return loss

def train_step(step, model, optimizer, lr_scheduler, train_dataloader, loss_function, diffusion_process, args, autoencoder=None):
    model.train().to(DEVICE)
    # TODO: check why data type changes from float64 to DoubleTensor somewhere here...    
    data, label = next(train_dataloader)
    data = data.float().to(DEVICE)
    label = label.to(DEVICE)

    if args.model.name == "gfdm-unet-1d-cond":
        if data.ndim == 2:
            data = data[:, None, :]
        label = label.float()
        if autoencoder is not None:
            with torch.no_grad():
                z = autoencoder.encoder(data.squeeze(1))
            data = z[:, None, :]
        orig_len = data.shape[-1]
        factor = getattr(model, "downsample_factor", 1)
        if autoencoder is not None:
            if orig_len % factor != 0:
                data, pad_len = pad_1d_to_factor(data, factor)
                if step == 0:
                    print(
                        f"Warning: latent length {orig_len} not divisible by {factor}; "
                        f"padding to {orig_len + pad_len}"
                    )
            else:
                pad_len = 0
                if step == 0:
                    print(f"Latent length {orig_len} divisible by {factor}; skipping padding")
        else:
            data, pad_len = pad_1d_to_factor(data, factor)
            if pad_len > 0 and step == 0:
                print(f"Padded input length from {orig_len} to {orig_len + pad_len} (factor {factor})")
    else:
        orig_len = None

    b, *_ = data.shape
    # sample a random timepoints for the backward process
    t = (torch.rand(b, device=data.device) * (args.diffusion.T - args.diffusion.eps) + args.diffusion.eps)

    noise = torch.randn_like(data, device=data.device)

    # run a backward SDE with this random timeline 
    loss = one_step_score_estimation(data, t, noise, label, model, loss_function, diffusion_process, args, orig_len=orig_len)

    # optimise the model
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    lr_scheduler.step()

    wandb.log({"train/loss": loss,
                "train/lr": lr_scheduler.get_last_lr()[0]},
                step=step)
    
def train(args):
    #torch.set_default_dtype(torch.float64)

    print("Training started...")
    print(f"Using the model type: {args.model.name}")

    ########################### AUTOENCODER TRAINING ###########################
    # get dataloaders once for autoencoder training and conditioning shape
    train_dataloader, valid_dataloader = get_dataloader(args)

    # for now a little brittle re-writing of args as every ROI and every ANN layer will have different dimensions 
    # and we need to set them here before we create the model;
    #  TODO: make it more elegant later
    if args.data.data_name == "ann-brain":
        one_fmri_signal, one_cond_signal = next(iter(train_dataloader))
        args.model.input_size = one_fmri_signal.shape[1]
        args.model.cross_attention_dim = one_cond_signal.shape[1]

    autoencoder = train_linear_autoencoder(args, train_dataloader, DEVICE)   
    if autoencoder is not None:
        autoencoder.eval()  

        # Log a sample reconstruction to wandb
        with torch.no_grad():
            sample_data, _ = next(iter(train_dataloader))
            sample_data = sample_data[0].float().to(DEVICE) # take the first element of the batch
            if sample_data.ndim == 3:
                sample_data = sample_data.squeeze(1)
            
            latent = autoencoder.encoder(sample_data)
            reconstructed = autoencoder.decoder(latent)
            
            sample_data = sample_data.cpu().squeeze().numpy()
            reconstructed = reconstructed.cpu().squeeze().numpy()

            #pyplot_brain(sample_data, args=args, savename="original_sample", figpath=f"{args.validation.output_folder}/{args.jobid}/autoencoder", save_type='png')
            #pyplot_brain(reconstructed, args=args, savename="reconstructed_sample", figpath=f"{args.validation.output_folder}/{args.jobid}/autoencoder", save_type='png')
            r2_scores_sample_recounstructed = pearsonr(sample_data, reconstructed)[0]
            wandb.log({"autoencoder/r2_score": r2_scores_sample_recounstructed}, step=0)
            print(f"Autoencoder sample reconstruction r2 score: {r2_scores_sample_recounstructed:.4f}", flush=True)

        args.model.input_size_original = args.model.input_size
        args.model.input_size = args.model.ae_latent_dim
        print(f"Autoencoder enabled: input_size set to latent dim {args.model.input_size}")
    ####################################################################################################
    
    # TODO: get the model if exists for continual training; probably will need to specify a path in the config file
    model = set_model(args)
    optimizer = set_optimiser(args, model)
    loss_function = set_loss_function(args)
    lr_scheduler = set_learning_rate_scheduler(optimizer, args)

    # re-create dataloaders for diffusion training
    # train_dataloader, valid_dataloader = get_dataloader(args)
    train_dataloader = itertools.cycle(train_dataloader)
    valid_dataloader = itertools.cycle(valid_dataloader)

    print(f"We're getting diffusion type {args.diffusion.diffusion_type}", flush=True)
    diffusion_process = diffusivity.get_diffusion(args, device=DEVICE)

    if args.diffusion.diffusion_type == "vp":
        print(f"beta min is {diffusion_process.beta_min}, beta_max is {diffusion_process.beta_max}", flush=True)
    elif args.diffusion.diffusion_type == "ve":
        print(f"sigma_min is {diffusion_process.sigma_min}, sigma_max is {diffusion_process.sigma_max}", flush=True)

    for step in range(args.train.steps):
        print(f"Step {step+1}/{args.train.steps} started.", flush=True)
        train_step(step, model, optimizer, lr_scheduler, train_dataloader, loss_function, diffusion_process, args, autoencoder=autoencoder)
        
        if step % args.validation.eval_freq == 0:
            print(f"Validation at step {step+1}", flush=True)
            # this one is a spcial case since we don't have a fixel label here like in MNIST case, 
            # rather continuous vectors that we should sample from the validation dataloader
            # TODO: move this correlation calculation to a separate function!
            if args.model.name in ["gfdm-unet-1d-cond", "dit"] and args.data.data_name == "ann-brain":
                true_fmri, cond = next(valid_dataloader)
                #cond = cond[:args.validation.batch_size].float().to(DEVICE)
                cond = cond.float().to(DEVICE)
                print("Generating sample of batch_size:", args.validation.batch_size, flush=True)
                generated_samples = diffusivity.generate_samples(
                    args.validation.batch_size, #cond.shape[0],
                    model,
                    diffusion_process,
                    args,
                    device=DEVICE,
                    cond=cond,
                )

                if autoencoder is not None:
                    with torch.no_grad():
                        z = generated_samples.squeeze(1)
                        generated_samples = autoencoder.decoder(z)
                else:
                    generated_samples = generated_samples.squeeze(1)  # remove channel dim
                
                # to store r2 scores across images for each voxel
                r2_scores_across_batch_images = np.empty((generated_samples.shape[1],))
                r2_scores_across_voxels = np.empty((generated_samples.shape[0],))

                print("True fmri shape:", true_fmri.shape, flush=True)
                print("Generated samples shape:", generated_samples.shape, flush=True)

                print("Calculating r2 scores across batch images...", flush=True)
                for voxel_idx in range(generated_samples.shape[1]):
                    true_fmri_by_image = true_fmri[:,voxel_idx].numpy()
                    generated_sample = generated_samples[:,voxel_idx].cpu().numpy()
                    assert len(true_fmri_by_image) == len(generated_sample)
                    r2_scores_across_batch_images[voxel_idx] = pearsonr(true_fmri_by_image, generated_sample)[0]

                print("Calculating r2 scores across voxels...", flush=True)
                for image_idx in range(generated_samples.shape[0]):
                    true_fmri_by_voxel = true_fmri[image_idx,:].numpy()
                    generated_sample = generated_samples[image_idx,:].cpu().numpy()
                    assert len(true_fmri_by_voxel) == len(generated_sample)
                    r2_scores_across_voxels[image_idx] = pearsonr(true_fmri_by_voxel, generated_sample)[0]
                
                visualise_and_save_results(generated_samples, step, args, r2_scores_across_batch_images=r2_scores_across_batch_images, r2_scores_across_voxels=r2_scores_across_voxels)

            else:
                generated_samples = diffusivity.generate_samples(args.validation.batch_size, model, diffusion_process, args, device=DEVICE)
                # TODO: add other image statistics later 
                visualise_and_save_results(generated_samples, step, args)

        # save the models throughout the training
        if step % args.model.save_freq == 0 and step > 0:
            directory_to_save = f"{args.model.output_folder}/{args.jobid}"
            if not os.path.exists(directory_to_save):
                    os.makedirs(directory_to_save)
            torch.save(model.state_dict(), f"{directory_to_save}/model_step_{step}.pth")        
            print(f"Model saved at step {step}.", flush=True)    
            
    # save the model
    directory_to_save = f"{args.model.output_folder}/{args.jobid}"
    if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
    torch.save(model.state_dict(), f"{directory_to_save}/model_final.pth")     
    print("Saved the final model.", flush=True)

    print("Training completed.", flush=True)

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args_and_setup_wandb()
    train(args)    

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()    
