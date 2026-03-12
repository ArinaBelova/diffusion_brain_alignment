import torch 
import torchvision
import wandb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import os
import torch.nn.functional as F
from contextlib import nullcontext
from scipy.stats import pearsonr

from diffusion_brain.utils.grad_updaters import set_loss_function, set_optimiser, set_learning_rate_scheduler 
from diffusion_brain.models import set_model
from diffusion_brain.models.autoencoder import LinearAutoencoder
from diffusion_brain.data_utils import get_dataloader
import diffusion_brain.utils.diffusivity as diffusivity
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results, pyplot_brain


def log_wandb(payload, step=None):
    if wandb.run is None:
        return
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def pad_1d_to_factor(x, factor):
    if factor <= 1:
        return x, 0
    length = x.shape[-1]
    pad_len = (factor - (length % factor)) % factor
    if pad_len == 0:
        return x, 0
    return F.pad(x, (0, pad_len)), pad_len


def pad_2d_to_multiple(x, multiple=64):
    """Pad 2D images to be divisible by multiple (e.g., 64 for 3 downsampling stages).
    
    Args:
        x: Tensor of shape (B, C, H, W)
        multiple: The spatial dimensions should be divisible by this value
    
    Returns:
        Padded tensor and tuple of (pad_h, pad_w) for later cropping
    """
    _, _, h, w = x.shape
    pad_h, pad_w = compute_2d_padding(h, w, multiple)
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    # F.pad format: (left, right, top, bottom)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, (pad_h, pad_w)


def compute_2d_padding(h, w, multiple=64):
    """Compute padding needed to make dimensions divisible by multiple."""
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    return pad_h, pad_w


def log_loss_histogram(losses, step, num_bins=100):
    """Create and log a histogram of loss values to wandb"""
    if len(losses) == 0:
        return
    
    losses_array = np.array(losses)
    
    # Create histogram data for wandb
    wandb_histogram = wandb.Histogram(sequence=losses_array, num_bins=num_bins)
    
    log_wandb(
        {
            "train/loss_histogram": wandb_histogram,
            "train/loss_stats": {
                "mean": float(np.mean(losses_array)),
                "min": float(np.min(losses_array)),
                "max": float(np.max(losses_array)),
            }
        },
        step=step,
    )


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
        log_wandb({"autoencoder/loss": avg_loss, "autoencoder/epoch": epoch + 1})

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


def one_step_score_estimation(x, t, noise, label, score_fn, loss_function, diffusion_process, args, orig_len=None, orig_shape=None, autocast_ctx=None):
    # TODO: check this! here I simply need to estimate p_{0t}(x(t)|x(0)) mean and variance and use them to compute the true score    
    true_score = -noise
    mu, std = diffusion_process.brown_moments(x, t)
    x_t = mu + std * noise

    #print("Type of the label is ", label.dtype, flush=True)
    mask = torch.bernoulli(torch.full((len(label),), float(args.model.dropout_prob))).to(label.device) 

    if args.data.data_name == "ann-brain":
        # for ann-brain data we have a continuous label vector representation instead of discrete num_classes
        masked_labels = label * (1 - mask[:, None])
        #print("Masked labels shape in score estimation: ", masked_labels.shape, flush=True)
        # Expand to sequence length for better cross-attention (repeat along seq dim)
        cond_seq_len = getattr(args.model, "cond_seq_len", 4)
        encoder_hidden_states = masked_labels.unsqueeze(1).expand(-1, cond_seq_len, -1).float()
        class_labels_arg = None
    else:    
        masked_labels = label * (1 - mask) + (args.model.num_classes * mask) # don't use -1 as the empty label as nn.Embedding will throw error
        masked_labels = masked_labels.long()    
        class_labels_arg = args.model.num_classes

    # Use autocast context for mixed precision if provided
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    
    with ctx:
        if args.model.name in ["unet-diffusers", "unet-diffusers-1d"]:
            #######################
            t = t * 999
            #######################

            if args.data.data_name == "mnist":
                encoder_hidden_states = torch.zeros(x_t.shape[0], 1, args.model.cross_attention_dim, device=x_t.device)

            predicted_score = score_fn(x_t, t, encoder_hidden_states = encoder_hidden_states, class_labels=class_labels_arg).sample
        elif args.model.name == "gfdm-unet-1d-cond":
            predicted_score = score_fn(x_t, t, masked_labels.float())
        else:
            predicted_score = score_fn(x_t, t, masked_labels)

        if orig_len is not None:
            # 1D case: crop to original length
            predicted_score = predicted_score[..., :orig_len]
            true_score = true_score[..., :orig_len]
        elif orig_shape is not None:
            # 2D case: crop to original spatial dimensions
            h, w = orig_shape
            predicted_score = predicted_score[..., :h, :w]
            true_score = true_score[..., :h, :w]

        loss = loss_function(predicted_score, true_score)
    
    return loss

def train_step(step, model, optimizer, lr_scheduler, train_dataloader, loss_function, diffusion_process, args, 
               autoencoder=None, loss_history=None, use_amp=False, accumulation_steps=1):
    """
    Single training step with gradient accumulation and mixed precision support.
    
    Args:
        step: Current global step
        model: The model to train
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        train_dataloader: Data iterator
        loss_function: Loss function
        diffusion_process: Diffusion process
        args: Training arguments
        autoencoder: Optional autoencoder for latent space
        loss_history: Optional list to track losses
        use_amp: Whether to use automatic mixed precision (bfloat16)
        accumulation_steps: Number of steps to accumulate gradients
    """
    model.train()
    
    # Mixed precision setup - create autocast context once
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if use_amp else None
    
    accumulated_loss = 0.0
    
    for accum_step in range(accumulation_steps):
        data, label = next(train_dataloader)
        data = data.float().to(DEVICE)
        label = label.to(DEVICE)
        
        ########## padding logic for UNet from diffusers and 1D GFDM UNet conditional ##########
        orig_shape = None  # Track original shape for 2D padding
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
                    if step == 0 and accum_step == 0:
                        print(
                            f"Warning: latent length {orig_len} not divisible by {factor}; "
                            f"padding to {orig_len + pad_len}"
                        )
                else:
                    pad_len = 0
                    if step == 0 and accum_step == 0:
                        print(f"Latent length {orig_len} divisible by {factor}; skipping padding")
            else:
                data, pad_len = pad_1d_to_factor(data, factor)
                if pad_len > 0 and step == 0 and accum_step == 0:
                    print(f"Padded input length from {orig_len} to {orig_len + pad_len} (factor {factor})")
        elif args.model.name == "unet-diffusers":
            # Pad 2D images to be divisible by 64 (supports up to 6 downsampling stages: 2^6 = 64)
            # For N stages, minimum multiple is 2^N (e.g., 4 stages → 16, but 64 is safe for all)
            orig_shape = data.shape[-2:]  # (H, W)
            data, (pad_h, pad_w) = pad_2d_to_multiple(data, multiple=64)
            if step == 0 and accum_step == 0:
                print(f"2D padding: {orig_shape} -> {data.shape[-2:]} (pad_h={pad_h}, pad_w={pad_w})")
            orig_len = None
        else:
            orig_len = None
        ##############################################################################################


        b, *_ = data.shape
        # sample random timepoints for the backward process
        t = (torch.rand(b, device=data.device) * (args.diffusion.T - args.diffusion.eps) + args.diffusion.eps)
        noise = torch.randn_like(data, device=data.device)

        # Forward pass with mixed precision
        loss = one_step_score_estimation(
            data, t, noise, label, model, loss_function, diffusion_process, args, 
            orig_len=orig_len, orig_shape=orig_shape, autocast_ctx=autocast_ctx
        )
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        accumulated_loss += loss.item() * accumulation_steps  # Unscale for logging

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    # Track loss for histogram
    if loss_history is not None:
        loss_history.append(accumulated_loss)

    log_wandb(
        {
            "train/loss": accumulated_loss,
            "train/lr": lr_scheduler.get_last_lr()[0],
        },
        step=step,
    )
    
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
        # #####################
        # grid_to_display = torchvision.utils.make_grid(one_fmri_signal * 255, nrow=int(np.sqrt(args.train.batch_size)))
        # wandb.log({"one_fmri_signal": wandb.Image(grid_to_display)})
        # #####################
        args.model.cross_attention_dim = one_cond_signal.shape[1]
        
        # For 2D data, we need to compute the padded size for UNet compatibility
        # Store original size for later cropping during generation
        original_shape = tuple(one_fmri_signal.shape[1:])  # (C, H, W)
        args.model.input_size_original = original_shape
        
        is_2d = getattr(args.data, "is_2d", False)
        if is_2d and len(original_shape) == 3:
            c, h, w = original_shape
            pad_h, pad_w = compute_2d_padding(h, w, multiple=64)
            args.model.input_size = (c, h + pad_h, w + pad_w)
            print(f"2D input size: original {original_shape} -> padded {args.model.input_size}")
        else:
            args.model.input_size = original_shape

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
            r_scores_sample_recounstructed = pearsonr(sample_data, reconstructed)[0]
            log_wandb({"autoencoder/r_score": r_scores_sample_recounstructed}, step=0)
            print(f"Autoencoder sample reconstruction r score: {r_scores_sample_recounstructed:.4f}", flush=True)

        args.model.input_size_original = args.model.input_size
        args.model.input_size = args.model.ae_latent_dim
        print(f"Autoencoder enabled: input_size set to latent dim {args.model.input_size}")
    ####################################################################################################

    # TODO: get the model if exists for continual training; probably will need to specify a path in the config file
    model = set_model(args).to(DEVICE)
    
    # Enable gradient checkpointing to reduce memory usage (trades compute for memory)
    use_gradient_checkpointing = getattr(args.train, "gradient_checkpointing", True)
    if use_gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled (saves ~30-40% memory)")

    optimizer = set_optimiser(args, model)
    loss_function = set_loss_function(args)
    
    # Gradient accumulation setup (must be before LR scheduler)
    accumulation_steps = getattr(args.train, "gradient_accumulation_steps", 1)
    effective_batch_size = args.train.batch_size * accumulation_steps
    print(f"Gradient accumulation: {accumulation_steps} steps, effective batch size: {effective_batch_size}")
    
    # LR scheduler - total_steps equals args.train.steps since we step once per optimizer update
    lr_scheduler = set_learning_rate_scheduler(optimizer, args, total_steps=args.train.steps)

    # Mixed precision training setup (bfloat16)
    use_amp = getattr(args.train, "use_mixed_precision", True) and DEVICE.type == "cuda"
    if use_amp:
        if torch.cuda.is_bf16_supported():
            print("Mixed precision training enabled with bfloat16")
        else:
            print("Warning: bfloat16 not supported on this device, falling back to float32")
            use_amp = False
    else:
        print("Mixed precision training disabled, using float32")

    train_iterator = infinite_loader(train_dataloader)
    valid_iterator = None
    if valid_dataloader is not None:
        valid_iterator = infinite_loader(valid_dataloader)

    print(f"We're getting diffusion type {args.diffusion.diffusion_type}", flush=True)
    diffusion_process = diffusivity.get_diffusion(args, device=DEVICE)

    if args.diffusion.diffusion_type == "vp":
        print(f"beta min is {diffusion_process.beta_min}, beta_max is {diffusion_process.beta_max}", flush=True)
    elif args.diffusion.diffusion_type == "ve":
        print(f"sigma_min is {diffusion_process.sigma_min}, sigma_max is {diffusion_process.sigma_max}", flush=True)

    # Track best validation metric for smart model saving
    best_val_r_score = -float('inf')
    
    for step in range(args.train.steps):
        print(f"Step {step+1}/{args.train.steps} started.", flush=True)

        loss_history = []
        train_step(
            step, model, optimizer, lr_scheduler, train_iterator, loss_function, 
            diffusion_process, args, autoencoder=autoencoder, loss_history=loss_history,
            use_amp=use_amp, accumulation_steps=accumulation_steps
        )

        # Log loss histogram periodically (e.g., every 100 steps)
        histogram_freq = getattr(args.train, "histogram_freq", 100)
        if step % histogram_freq == 0 and step > 0:
            log_loss_histogram(loss_history, step)
        
        if valid_iterator is not None and step % args.validation.eval_freq == 0:
            print(f"Validation at step {step+1}", flush=True)
            # Switch to eval mode for full precision inference (float32)
            model.eval()
            
            # Inference runs in full precision (no autocast)
            with torch.no_grad():
                # this one is a special case since we don't have a fixed label here like in MNIST case, 
                # rather continuous vectors that we should sample from the validation dataloader
                # TODO: move this correlation calculation to a separate function!
                current_val_r_score = None
                if args.data.data_name == "ann-brain":
                    true_fmri, cond = next(valid_iterator)
                    cond = cond.float().to(DEVICE)
                    print("Generating sample of batch_size:", args.validation.batch_size, flush=True)
                    generated_samples = diffusivity.generate_samples(
                        args.validation.batch_size,
                        model,
                        diffusion_process,
                        args,
                        device=DEVICE,
                        cond=cond,
                    )

                    if autoencoder is not None:
                        z = generated_samples.squeeze(1)
                        generated_samples = autoencoder.decoder(z)
                    else:
                        generated_samples = generated_samples.squeeze(1)  # remove channel dim

                    current_val_r_score = visualise_and_save_results(generated_samples, true_fmri, step, args)
                else: # toy, mnist and other data with discrete labels  
                    generated_samples = diffusivity.generate_samples(args.validation.batch_size, model, diffusion_process, args, device=DEVICE)
                    # TODO: add other image statistics later 
                    print("Generated samples shape:", generated_samples.shape, flush=True)
                    visualise_and_save_results(generated_samples, step, args)
            
            # Save best model based on validation r-score
            if current_val_r_score is not None and current_val_r_score > best_val_r_score:
                best_val_r_score = current_val_r_score
                directory_to_save = f"{args.model.output_folder}/{args.jobid}"
                if not os.path.exists(directory_to_save):
                    os.makedirs(directory_to_save)
                torch.save(model.state_dict(), f"{directory_to_save}/model_best.pth")
                print(f"New best model saved at step {step} with r-score: {best_val_r_score:.4f}", flush=True)
                log_wandb({"validation/best_r_score": best_val_r_score, "validation/best_step": step}, step=step)

        # Periodic checkpoint saving (in addition to best model saving)
        if step % args.model.save_freq == 0 and step > 0:
            directory_to_save = f"{args.model.output_folder}/{args.jobid}"
            if not os.path.exists(directory_to_save):
                    os.makedirs(directory_to_save)
            torch.save(model.state_dict(), f"{directory_to_save}/model_checkpoint_step_{step}.pth")
            print(f"Checkpoint saved at step {step}.", flush=True)    
            
    # save the model
    directory_to_save = f"{args.model.output_folder}/{args.jobid}"
    if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
    torch.save(model.state_dict(), f"{directory_to_save}/model_final.pth") # args.train.steps instead of final ?
    print("Saved the final model.", flush=True)
    print("Training completed.", flush=True)

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args_and_setup_wandb(init_wandb=True)
    print("ARGS: ", args, flush=True)

    global DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    try:
        train(args)
    finally:
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()    
