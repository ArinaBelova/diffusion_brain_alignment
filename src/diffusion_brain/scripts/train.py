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

from diffusion_brain.utils.grad_updaters import set_loss_function, set_optimiser, set_learning_rate_scheduler, EMAModel
from diffusion_brain.models import set_model, ANNTokenizer
from diffusion_brain.models.autoencoder import LinearAutoencoder
from diffusion_brain.data_utils import get_dataloader
import diffusion_brain.utils.diffusivity as diffusivity
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results, pyplot_brain

# from diffusion_brain.debug_cross_attention import run_all_diagnostics

class CrossAttnDiagnostics:
    """Monkey-patches CrossAttn layers to directly capture conditioning usage (bypassing hook limitations)."""
    
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        self.attention_data = {}
        self.num_layers_patched = 0
        self.register_patches()
    
    def register_patches(self):
        """Monkey-patch attn2 Attention layers to capture encoder_hidden_states directly."""
        found_layers = []
        
        for name, module in self.model.named_modules():
            # Look for the actual Attention modules inside attn2
            if "attn2" in name and type(module).__name__ == "Attention":
                try:
                    # Save original forward method
                    self.original_forwards[name] = module.forward
                    
                    # Create patched forward method with access to kwargs
                    patched_forward = self._make_patched_forward(name, module, self.original_forwards[name])
                    module.forward = patched_forward
                    
                    self.num_layers_patched += 1
                    found_layers.append(name)
                except Exception as e:
                    print(f"  - Failed to patch {name}: {e}", flush=True)
        
        if found_layers:
            print(f"Attention modules in attn2 found and patched: {len(found_layers)} total", flush=True)
            for layer in found_layers[:3]:  # Print first 3
                print(f"  - {layer}", flush=True)
            if len(found_layers) > 3:
                print(f"  ... and {len(found_layers)-3} more", flush=True)
        else:
            print("WARNING: No attn2 Attention modules found in model!", flush=True)
        
        print(f"Total Attention patches registered: {self.num_layers_patched}", flush=True)
    
    def _make_patched_forward(self, name, module, original_forward):
        """Create a patched forward method that captures kwargs and attention dynamics."""
        def patched_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # CAPTURE: encoder_hidden_states statistics
            if encoder_hidden_states is not None:
                self.attention_data[f"{name}_hidden_mean"] = float(encoder_hidden_states.mean().detach().item())
                self.attention_data[f"{name}_hidden_std"] = float(encoder_hidden_states.std().detach().item())
                self.attention_data[f"{name}_hidden_norm"] = float(torch.norm(encoder_hidden_states).detach().item())
                self.attention_data[f"{name}_hidden_shape"] = str(encoder_hidden_states.shape)
                self.attention_data[f"{name}_has_conditioning"] = True
            else:
                self.attention_data[f"{name}_has_conditioning"] = False
            
            # CAPTURE: hidden_states (query) statistics
            self.attention_data[f"{name}_hidden_in_norm"] = float(torch.norm(hidden_states).detach().item())
            
            # Call original forward and capture output
            output = original_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, 
                                     attention_mask=attention_mask, **kwargs)
            
            # CAPTURE: output statistics to see if conditioning affected it
            if isinstance(output, torch.Tensor):
                self.attention_data[f"{name}_output_norm"] = float(torch.norm(output).detach().item())
            
            return output
        
        return patched_forward
    
    def get_diagnostics(self):
        """Return collected diagnostics and reset."""
        diag = dict(self.attention_data)
        self.attention_data = {}
        return diag
    
    def cleanup(self):
        """Restore original forward methods."""
        for name, original_forward in self.original_forwards.items():
            try:
                for mod_name, module in self.model.named_modules():
                    if mod_name == name:
                        module.forward = original_forward
                        break
            except Exception as e:
                print(f"Failed to restore {name}: {e}", flush=True)


class AttentionWeightsDiagnostics:
    """Hooks into actual attention computation to capture conditioning sensitivity."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_data = {}
        self.outputs_real = {}  # Store outputs from real conditioning pass
        self.outputs_shuffled = {}  # Store outputs from shuffled conditioning pass
        self.pass_mode = "real"  # Track which pass we're in
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks on Attention.__call__ to capture attention outputs."""
        found_layers = []
        
        for name, module in self.model.named_modules():
            if "attn2" in name and type(module).__name__ == "Attention":
                try:
                    # Register forward hook to capture outputs including attention weights
                    hook = module.register_forward_hook(self._make_output_hook(name))
                    self.hooks.append(hook)
                    found_layers.append(name)
                except Exception as e:
                    print(f"  - Failed to register attention hook on {name}: {e}", flush=True)
        
        if found_layers:
            print(f"Attention sensitivity hooks registered on {len(found_layers)} layers", flush=True)
    
    def _make_output_hook(self, name):
        """Create a hook to capture attention output for sensitivity analysis."""
        def hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    # Store output norm for this pass
                    output_norm = float(torch.norm(output).detach().item())
                    
                    if self.pass_mode == "real":
                        self.outputs_real[name] = output_norm
                    elif self.pass_mode == "shuffled":
                        self.outputs_shuffled[name] = output_norm
            except Exception as e:
                pass  # Silently fail to not clutter logs
        
        return hook
    
    def set_pass_mode(self, mode):
        """Set whether we're capturing real or shuffled conditioning pass."""
        self.pass_mode = mode  # "real" or "shuffled"
    
    def compute_sensitivity(self):
        """Compute how different outputs are between real and shuffled conditioning."""
        sensitivity_data = {}
        
        for name in self.outputs_real.keys():
            if name in self.outputs_shuffled:
                real_norm = self.outputs_real[name]
                shuffled_norm = self.outputs_shuffled[name]
                
                # Compute difference as indicator of conditioning sensitivity
                if real_norm + shuffled_norm > 0:
                    relative_diff = abs(real_norm - shuffled_norm) / (real_norm + shuffled_norm)
                    sensitivity_data[f"{name}_cond_sensitivity"] = relative_diff
        
        # Clear buffers
        self.outputs_real = {}
        self.outputs_shuffled = {}
        
        return sensitivity_data
    
    def get_diagnostics(self):
        """Return collected diagnostics."""
        diag = dict(self.attention_data)
        self.attention_data = {}
        return diag
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()


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


def one_step_score_estimation(x, t, noise, label, score_fn, loss_function, diffusion_process, args, step=0, orig_len=None, orig_shape=None, autocast_ctx=None, attn_weights_diag=None, ann_tokenizer=None):
    # TODO: check this! here I simply need to estimate p_{0t}(x(t)|x(0)) mean and variance and use them to compute the true score    
    true_score = -noise
    mu, std = diffusion_process.brown_moments(x, t)
    x_t = mu + std * noise

    #print("Type of the label is ", label.dtype, flush=True)
    mask = torch.bernoulli(torch.full((len(label),), float(args.model.dropout_prob))).to(label.device) 

    def to_cond_tokens(cond_vec):
        """Tokenize ANN vector for cross-attention encoder_hidden_states."""
        if ann_tokenizer is not None:
            return ann_tokenizer(cond_vec.float())
        # Legacy fallback for chunk/repeat modes
        cond_seq_len = max(1, int(getattr(args.model, "cond_seq_len", 1)))
        cond_token_mode = getattr(args.model, "cond_token_mode", "repeat")
        if cond_token_mode == "chunk" and cond_seq_len > 1 and cond_vec.shape[1] % cond_seq_len == 0:
            return cond_vec.reshape(cond_vec.shape[0], cond_seq_len, -1).float()
        return cond_vec.unsqueeze(1).expand(-1, cond_seq_len, -1).float()

    if args.data.data_name == "ann-brain":
        # for ann-brain data we have a continuous label vector representation instead of discrete num_classes
        masked_labels = label * (1 - mask[:, None])
        encoder_hidden_states = to_cond_tokens(masked_labels)

        # class_labels is NOT used for ANN conditioning — that slot is reserved
        # for future discrete subject-identity embedding
        class_labels_arg = None

        # Log encoder_hidden_states diagnostics every 100 steps
        # if step % 100 == 0:
        #     print(
        #         f"[step {step}] encoder_hidden_states: "
        #         f"shape={encoder_hidden_states.shape}, "
        #         f"mean={encoder_hidden_states.mean().item():.6f}, "
        #         f"std={encoder_hidden_states.std().item():.6f}, "
        #         f"norm={torch.norm(encoder_hidden_states).item():.6f}, "
        #         f"min={encoder_hidden_states.min().item():.6f}, "
        #         f"max={encoder_hidden_states.max().item():.6f}",
        #         flush=True
        #     )
    else:
        masked_labels = label * (1 - mask) + (args.model.num_classes * mask) # don't use -1 as the empty label as nn.Embedding will throw error
        masked_labels = masked_labels.long()
        class_labels_arg = args.model.num_classes

    # Use autocast context for mixed precision if provided
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    
    # with ctx:
    if args.model.name in ["unet-diffusers", "unet-diffusers-1d"]:
        #######################
        t = t * 999
        #######################

        if args.data.data_name == "mnist":
            encoder_hidden_states = torch.zeros(x_t.shape[0], 1, args.model.cross_attention_dim, device=x_t.device)

        # Debug: log what we're passing to the model
        # if step % 100 == 0:
        #     print(
        #         f"[step {step}] Calling UNet with encoder_hidden_states: "
        #         f"type={type(encoder_hidden_states)}, "
        #         f"shape={encoder_hidden_states.shape if hasattr(encoder_hidden_states, 'shape') else 'N/A'}, "
        #         f"is_none={encoder_hidden_states is None}",
        #         flush=True
        #     )
        
        predicted_score = score_fn(x_t, t, encoder_hidden_states = encoder_hidden_states, class_labels=class_labels_arg).sample
    elif args.model.name == "gfdm-unet-1d-cond":
        cond_mode = getattr(args.model, "cond_mode", "additive")
        if cond_mode == "cross_attention" and args.data.data_name == "ann-brain":
            # Cross-attention: tokenize ANN → (B, num_tokens, token_dim) → permute to (B, token_dim, num_tokens) for conv1d encoder_kv
            encoder_out = encoder_hidden_states.permute(0, 2, 1).contiguous()
            predicted_score = score_fn(x_t, t, encoder_out=encoder_out)
        else:
            # Additive conditioning: pass raw ANN vector directly
            predicted_score = score_fn(x_t, t, cond=masked_labels.float())
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

    denoise_loss = loss_function(predicted_score, true_score)

    # Conditioning separation loss (ann-brain only):
    # Penalise the model when the conditional and unconditional predictions are
    # identical. Without this, the backbone learns a "good enough" unconditional
    # score and gradients through the cross-attention pathway vanish.
    # lambda_cond_sep controls how strongly conditioning is encouraged; set to
    # 0.0 in the config to disable.
    lambda_cond_sep = float(getattr(args.model, "lambda_cond_sep", 0.0))
    _use_cross_attn_1d = args.model.name == "gfdm-unet-1d-cond" and getattr(args.model, "cond_mode", "additive") == "cross_attention"
    if lambda_cond_sep > 0.0 and args.data.data_name == "ann-brain" and (args.model.name in ["unet-diffusers", "unet-diffusers-1d"] or _use_cross_attn_1d):
        cond_tokens_sep = to_cond_tokens(label.float())
        uncond_tokens_sep = torch.zeros_like(cond_tokens_sep)
        # Detach x_t so backbone ResNet blocks receive no gradient from sep_loss.
        # Only K/V projections (cross-attn) and ANNTokenizer get gradient,
        # preventing the backbone from learning to cancel the signal.
        x_t_sep = x_t.detach()
        if _use_cross_attn_1d:
            cond_enc = cond_tokens_sep.permute(0, 2, 1).contiguous()
            uncond_enc = uncond_tokens_sep.permute(0, 2, 1).contiguous()
            pred_cond_sep = score_fn(x_t_sep, t, encoder_out=cond_enc)
            pred_uncond_sep = score_fn(x_t_sep, t, encoder_out=uncond_enc)
        else:
            pred_cond_sep = score_fn(x_t_sep, t, encoder_hidden_states=cond_tokens_sep, class_labels=None).sample
            pred_uncond_sep = score_fn(x_t_sep, t, encoder_hidden_states=uncond_tokens_sep, class_labels=None).sample
        if orig_len is not None:
            pred_cond_sep = pred_cond_sep[..., :orig_len]
            pred_uncond_sep = pred_uncond_sep[..., :orig_len]
        elif orig_shape is not None:
            h, w = orig_shape
            pred_cond_sep = pred_cond_sep[..., :h, :w]
            pred_uncond_sep = pred_uncond_sep[..., :h, :w]
        # Maximise L1 separation between conditional and unconditional predictions.
        # Negated because we want to maximise, not minimise.
        sep_loss = -F.l1_loss(pred_cond_sep, pred_uncond_sep, reduction="mean")
        total_loss = denoise_loss + lambda_cond_sep * sep_loss
    else:
        total_loss = denoise_loss

    # Diagnostics: check if conditioning is actually affecting predictions
    # cond_diagnostics = {}
    # if step % 100 == 0:
    #     with torch.no_grad():
    #         if args.data.data_name == "ann-brain" and args.model.name == "gfdm-unet-1d-cond" and label.shape[0] > 1 and ann_tokenizer is not None:
    #             # 1D cross-attention conditioning diagnostics
    #             cond_tokens = to_cond_tokens(label.float())
    #             uncond_tokens = torch.zeros_like(cond_tokens)
    #             perm = torch.randperm(label.shape[0], device=label.device)
    #             shuffled_tokens = to_cond_tokens(label[perm].float())

    #             with torch.autocast(device_type="cuda", enabled=False):
    #                 x_t_f32 = x_t.float()
    #                 cond_out = cond_tokens.float().permute(0, 2, 1)
    #                 uncond_out = uncond_tokens.float().permute(0, 2, 1)
    #                 shuf_out = shuffled_tokens.float().permute(0, 2, 1)

    #                 cond_pred = score_fn(x_t_f32, t, encoder_out=cond_out)
    #                 shuffled_pred = score_fn(x_t_f32, t, encoder_out=shuf_out)
    #                 uncond_pred = score_fn(x_t_f32, t, encoder_out=uncond_out)

    #             if orig_len is not None:
    #                 cond_pred = cond_pred[..., :orig_len]
    #                 uncond_pred = uncond_pred[..., :orig_len]
    #                 shuffled_pred = shuffled_pred[..., :orig_len]

    #             sep_cond_uncond = F.l1_loss(cond_pred, uncond_pred, reduction="mean").item()
    #             sep_cond_shuffled = F.l1_loss(cond_pred, shuffled_pred, reduction="mean").item()
    #             sep_uncond_shuffled = F.l1_loss(uncond_pred, shuffled_pred, reduction="mean").item()

    #             cond_diagnostics.update({
    #                 "sep_cond_uncond": sep_cond_uncond,
    #                 "sep_cond_shuffled": sep_cond_shuffled,
    #                 "sep_uncond_shuffled": sep_uncond_shuffled,
    #                 "cond_pred_norm": float(torch.norm(cond_pred).item()),
    #                 "uncond_pred_norm": float(torch.norm(uncond_pred).item()),
    #                 "shuffled_pred_norm": float(torch.norm(shuffled_pred).item()),
    #             })
    #         elif args.data.data_name == "ann-brain" and args.model.name in ["unet-diffusers", "unet-diffusers-1d"] and label.shape[0] > 1:
    #             # Compute predictions with different conditioning inputs (ann-brain uses continuous labels)
    #             cond_tokens = to_cond_tokens(label.float())
    #             uncond_tokens = torch.zeros_like(cond_tokens)
    #             perm = torch.randperm(label.shape[0], device=label.device)
    #             shuffled_tokens = to_cond_tokens(label[perm].float())

    #             # Run in float32: bfloat16 precision (~0.8%) rounds away the
    #             # conditioning difference (<0.1%) making sep_cond_uncond appear zero.
    #             with torch.autocast(device_type="cuda", enabled=False):
    #                 x_t_f32 = x_t.float()
    #                 # REAL CONDITIONING PASS
    #                 if attn_weights_diag is not None:
    #                     attn_weights_diag.set_pass_mode("real")
    #                 cond_pred = score_fn(x_t_f32, t, encoder_hidden_states=cond_tokens.float(), class_labels=None).sample

    #                 # SHUFFLED CONDITIONING PASS
    #                 if attn_weights_diag is not None:
    #                     attn_weights_diag.set_pass_mode("shuffled")
    #                 shuffled_pred = score_fn(x_t_f32, t, encoder_hidden_states=shuffled_tokens.float(), class_labels=None).sample

    #                 # UNCOND PASS
    #                 uncond_pred = score_fn(x_t_f32, t, encoder_hidden_states=uncond_tokens.float(), class_labels=None).sample
                
    #             # Compute attention sensitivity (difference between real and shuffled conditioning)
    #             if attn_weights_diag is not None:
    #                 sensitivity = attn_weights_diag.compute_sensitivity()
    #                 cond_diagnostics.update(sensitivity)
                
    #             if orig_shape is not None:
    #                 h, w = orig_shape
    #                 cond_pred = cond_pred[..., :h, :w]
    #                 uncond_pred = uncond_pred[..., :h, :w]
    #                 shuffled_pred = shuffled_pred[..., :h, :w]
                
    #             # Compute separations
    #             sep_cond_uncond = F.l1_loss(cond_pred, uncond_pred, reduction="mean").item()
    #             sep_cond_shuffled = F.l1_loss(cond_pred, shuffled_pred, reduction="mean").item()
    #             sep_uncond_shuffled = F.l1_loss(uncond_pred, shuffled_pred, reduction="mean").item()
                
    #             cond_diagnostics.update({
    #                 "sep_cond_uncond": sep_cond_uncond,
    #                 "sep_cond_shuffled": sep_cond_shuffled,
    #                 "sep_uncond_shuffled": sep_uncond_shuffled,
    #                 "cond_pred_norm": float(torch.norm(cond_pred).item()),
    #                 "uncond_pred_norm": float(torch.norm(uncond_pred).item()),
    #                 "shuffled_pred_norm": float(torch.norm(shuffled_pred).item()),
    #             })
    #         elif args.data.data_name == "mnist" and args.model.name in ["unet-diffusers", "unet-diffusers-1d"] and label.shape[0] > 1:
    #             # Compute predictions with different discrete label conditioning (MNIST)
    #             cond_labels = label.long()
    #             uncond_labels = torch.full_like(cond_labels, args.model.num_classes)  # "empty" label
    #             perm = torch.randperm(label.shape[0], device=label.device)
    #             shuffled_labels = cond_labels[perm]
                
    #             # All use zeros for encoder_hidden_states (as in training)
    #             encoder_hidden_states_zeros = torch.zeros(x_t.shape[0], 1, args.model.cross_attention_dim, device=x_t.device)
                
    #             cond_pred = score_fn(x_t, t, encoder_hidden_states=encoder_hidden_states_zeros, class_labels=cond_labels).sample
    #             uncond_pred = score_fn(x_t, t, encoder_hidden_states=encoder_hidden_states_zeros, class_labels=uncond_labels).sample
    #             shuffled_pred = score_fn(x_t, t, encoder_hidden_states=encoder_hidden_states_zeros, class_labels=shuffled_labels).sample
                
    #             if orig_shape is not None:
    #                 h, w = orig_shape
    #                 cond_pred = cond_pred[..., :h, :w]
    #                 uncond_pred = uncond_pred[..., :h, :w]
    #                 shuffled_pred = shuffled_pred[..., :h, :w]
                
    #             # Compute separations
    #             sep_cond_uncond = F.l1_loss(cond_pred, uncond_pred, reduction="mean").item()
    #             sep_cond_shuffled = F.l1_loss(cond_pred, shuffled_pred, reduction="mean").item()
    #             sep_uncond_shuffled = F.l1_loss(uncond_pred, shuffled_pred, reduction="mean").item()
                
    #             cond_diagnostics = {
    #                 "sep_cond_uncond": sep_cond_uncond,
    #                 "sep_cond_shuffled": sep_cond_shuffled,
    #                 "sep_uncond_shuffled": sep_uncond_shuffled,
    #                 "cond_pred_norm": float(torch.norm(cond_pred).item()),
    #                 "uncond_pred_norm": float(torch.norm(uncond_pred).item()),
    #                 "shuffled_pred_norm": float(torch.norm(shuffled_pred).item()),
    #             }
    cond_diagnostics={}

    return total_loss, {
        "denoise_loss": float(denoise_loss.detach().item()),
        **cond_diagnostics,
    }

def save_checkpoint(directory, model, optimizer, lr_scheduler, step, best_val_r_score, ann_tokenizer=None, ema=None, suffix="", fmri_min=None, fmri_max=None):
    """Save a full training checkpoint that can be used to resume training.

    Args:
        directory: Directory to save checkpoint files
        model: The UNet model
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        step: Current training step
        best_val_r_score: Best validation r-score so far
        ann_tokenizer: Optional ANNTokenizer module
        ema: Optional EMAModel for exponential moving average weights
        suffix: Filename suffix (e.g., "best", "step_5000", "final")
    """
    os.makedirs(directory, exist_ok=True)
    checkpoint = {
        "step": step,
        "best_val_r_score": best_val_r_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }
    if ann_tokenizer is not None:
        checkpoint["ann_tokenizer_state_dict"] = ann_tokenizer.state_dict()
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    # Save normalised fMRI data range for thresholding at generation time
    if fmri_min is not None and fmri_max is not None:
        checkpoint["fmri_min"] = fmri_min
        checkpoint["fmri_max"] = fmri_max

    fname = f"checkpoint_{suffix}.pth" if suffix else "checkpoint.pth"
    torch.save(checkpoint, os.path.join(directory, fname))
    print(f"Checkpoint saved: {fname} at step {step}", flush=True)


def load_checkpoint(path, model, optimizer, lr_scheduler, ann_tokenizer=None, ema=None, device="cpu"):
    """Load a training checkpoint and restore all training state.

    Args:
        path: Path to the checkpoint file
        model: The UNet model (must already be created with matching architecture)
        optimizer: Optimizer (must already be created)
        lr_scheduler: Learning rate scheduler (must already be created)
        ann_tokenizer: Optional ANNTokenizer module
        ema: Optional EMAModel to restore
        device: Device to map tensors to

    Returns:
        step: The step to resume from (next step after the saved one)
        best_val_r_score: Best validation r-score at time of checkpoint
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    if ann_tokenizer is not None and "ann_tokenizer_state_dict" in checkpoint:
        ann_tokenizer.load_state_dict(checkpoint["ann_tokenizer_state_dict"])

    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        print(f"EMA state restored (num_updates={ema.num_updates})", flush=True)

    step = checkpoint["step"]
    best_val_r_score = checkpoint.get("best_val_r_score", -float('inf'))

    print(f"Checkpoint loaded from {path}: resuming from step {step + 1}, best_val_r_score={best_val_r_score:.4f}", flush=True)
    return step + 1, best_val_r_score


def train_step(step, model, optimizer, lr_scheduler, train_dataloader, loss_function, diffusion_process, args,
               autoencoder=None, loss_history=None, use_amp=False, accumulation_steps=1, crossattn_diag=None, attn_weights_diag=None,
               ann_tokenizer=None, ema=None):
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
        crossattn_diag: Optional CrossAttnDiagnostics for monitoring
    """    
    model.train()
    
    # Mixed precision setup - create autocast context once
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if use_amp else None
    
    accumulated_loss = 0.0
    accumulated_denoise_loss = 0.0
    accumulated_diagnostics = {}
    
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
            data, (pad_h, pad_w) = pad_2d_to_multiple(data, multiple=args.data.resize_to_multiple)

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
        loss, loss_metrics = one_step_score_estimation(
            data, t, noise, label, model, loss_function, diffusion_process, args,
            step=step, orig_len=orig_len, orig_shape=orig_shape, autocast_ctx=autocast_ctx,
            attn_weights_diag=attn_weights_diag, ann_tokenizer=ann_tokenizer
        )
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        accumulated_loss += loss.item() * accumulation_steps  # Unscale for logging
        accumulated_denoise_loss += loss_metrics["denoise_loss"]
        for key, val in loss_metrics.items():
            if key != "denoise_loss" and key not in accumulated_diagnostics:
                accumulated_diagnostics[key] = val

    # Collect gradient diagnostics from attn2 (cross-attention) layers BEFORE zero_grad
    # crossattn_grads = {}
    # layers_receiving_cond = 0
    # layers_not_receiving_cond = 0
    
    # if step % 100 == 0:
    #     # Direct parameter inspection for attn2 layers
    #     for name, param in model.named_parameters():
    #         if param.grad is not None and "attn2" in name:
    #             grad_norm = float(torch.norm(param.grad).item())
    #             if grad_norm > 0:
    #                 # Shorten name for logging: keep last 3 parts (needed for Q-norm Sequential)
    #                 parts = name.split(".")
    #                 short_name = ".".join(parts[-3:]) if len(parts) > 2 else ".".join(parts[-2:]) if len(parts) > 1 else parts[-1]
    #                 crossattn_grads[f"grad_attn2_{short_name}"] = grad_norm
        
    #     # Also collect from forward hook data if available
    #     if crossattn_diag is not None and crossattn_diag.num_layers_patched > 0:
    #         try:
    #             crossattn_diags = crossattn_diag.get_diagnostics()
    #             accumulated_diagnostics.update(crossattn_diags)
                
    #             # Count how many layers are receiving encoder_hidden_states (monkey-patch approach)
    #             for key, val in crossattn_diags.items():
    #                 if key.endswith("_has_conditioning"):
    #                     if val:
    #                         layers_receiving_cond += 1
    #                     else:
    #                         layers_not_receiving_cond += 1
    #         except Exception as e:
    #             print(f"Error collecting CrossAttn hook diagnostics: {e}", flush=True)
        
    #     # Add gradient info
    #     if crossattn_grads:
    #         accumulated_diagnostics.update(crossattn_grads)
    #         print(
    #             f"[step {step}] attn2 gradients: {len(crossattn_grads)} params with gradients, "
    #             f"layers with conditioning: {layers_receiving_cond}, without: {layers_not_receiving_cond}",
    #             flush=True
    #         )
    
    # gradient clipping: 
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)

    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    # Update EMA shadow weights after each optimizer step
    if ema is not None:
        all_params = list(model.parameters())
        if ann_tokenizer is not None:
            all_params += list(ann_tokenizer.parameters())
        ema.update(all_params)

    # Track loss for histogram
    if loss_history is not None:
        loss_history.append(accumulated_loss)

    avg_denoise_loss = accumulated_denoise_loss / max(1, accumulation_steps)

    payload = {
        "train/loss": accumulated_loss,
        "train/lr": lr_scheduler.get_last_lr()[0],
        "train/denoise_loss": avg_denoise_loss,
    }

    # Log any diagnostics collected
    for key, val in accumulated_diagnostics.items():
        if val is not None:
            payload[f"train/{key}"] = val

    log_wandb(
        payload,
        step=step,
    )
    
def train(args):
    #torch.set_default_dtype(torch.float64)

    print("Training started...")
    print(f"Using the model type: {args.model.name}")

    ########################### AUTOENCODER TRAINING ###########################
    # get dataloaders once for autoencoder training and conditioning shape
    train_dataloader, valid_dataloader = get_dataloader(args)

    # Store normalised fMRI data range for thresholding during generation
    train_ds = train_dataloader.dataset
    args.data.fmri_min = getattr(train_ds, "fmri_min", None)
    args.data.fmri_max = getattr(train_ds, "fmri_max", None)

    # for now a little brittle re-writing of args as every ROI and every ANN layer will have different dimensions
    # and we need to set them here before we create the model;
    #  TODO: make it more elegant later
    if args.data.data_name == "ann-brain":
        one_fmri_signal, one_cond_signal = next(iter(train_dataloader))
        # #####################
        # grid_to_display = torchvision.utils.make_grid(one_fmri_signal * 255, nrow=int(np.sqrt(args.train.batch_size)))
        # wandb.log({"one_fmri_signal": wandb.Image(grid_to_display)})
        # #####################
        ann_dim = one_cond_signal.shape[1]
        args.model.ann_dim = ann_dim
        cond_token_mode = getattr(args.model, "cond_token_mode", "learned")
        num_tokens = max(1, int(getattr(args.model, "cond_seq_len", 8)))
        token_dim = int(getattr(args.model, "token_dim", 256))

        if cond_token_mode == "learned":
            args.model.cross_attention_dim = token_dim
            print(
                f"Condition tokenization: LEARNED | ANN dim {ann_dim} -> "
                f"ANNTokenizer({num_tokens} tokens x {token_dim}-dim), "
                f"cross_attention_dim={token_dim}",
                flush=True,
            )
        else:
            # Legacy modes kept for backward compat / ablations
            cond_seq_len = num_tokens
            if cond_token_mode == "chunk" and cond_seq_len > 1 and ann_dim % cond_seq_len == 0:
                args.model.cross_attention_dim = ann_dim // cond_seq_len
                print(
                    f"Condition tokenization: chunk | ANN dim {ann_dim} -> "
                    f"seq_len {cond_seq_len} x token_dim {args.model.cross_attention_dim}",
                    flush=True,
                )
            else:
                args.model.cross_attention_dim = ann_dim
                args.model.cond_token_mode = "repeat"
                print(
                    f"Condition tokenization: repeat | seq_len {cond_seq_len}, "
                    f"token_dim {args.model.cross_attention_dim}",
                    flush=True,
                )
        
        # For 2D data, we need to compute the padded size for UNet compatibility
        # Store original size for later cropping during generation
        original_shape = tuple(one_fmri_signal.shape[1:])  # (C, H, W)
        args.model.input_size_original = original_shape
        
        is_2d = getattr(args.data, "is_2d", False)
        if is_2d and len(original_shape) == 3:
            c, h, w = original_shape
            pad_h, pad_w = compute_2d_padding(h, w, multiple=args.data.resize_to_multiple)
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

    # Create ANNTokenizer for learned conditioning (replaces manual chunk/repeat)
    ann_tokenizer = None
    cond_token_mode = getattr(args.model, "cond_token_mode", "learned")
    _1d_cross_attn = args.model.name == "gfdm-unet-1d-cond" and getattr(args.model, "cond_mode", "additive") == "cross_attention"
    if args.data.data_name == "ann-brain" and (args.model.name == "unet-diffusers" or _1d_cross_attn) and cond_token_mode == "learned":
        ann_dim = args.model.ann_dim
        num_tokens = max(1, int(getattr(args.model, "cond_seq_len", 8)))
        token_dim = int(getattr(args.model, "token_dim", 256))
        ann_tokenizer = ANNTokenizer(ann_dim=ann_dim, num_tokens=num_tokens, token_dim=token_dim).to(DEVICE)
        print(f"ANNTokenizer created: {ann_dim} -> {num_tokens} tokens x {token_dim}-dim")

    # Initialize CrossAttn diagnostics
    # crossattn_diag = None
    # attn_weights_diag = None
    # if args.data.data_name == "ann-brain" and args.model.name == "unet-diffusers":
    #     crossattn_diag = CrossAttnDiagnostics(model)
    #     print("CrossAttn diagnostics enabled - will track gradient flow and attention patterns")
    #     attn_weights_diag = AttentionWeightsDiagnostics(model)
    #     print("Attention weights diagnostics enabled - will track attention output diversity")
    crossattn_diag, attn_weights_diag = None, None  # Disabled for now to save overhead; can re-enable with config flag later

    # Log parameter counts
    def _count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    unet_params = _count_params(model)
    tok_params = _count_params(ann_tokenizer) if ann_tokenizer is not None else 0
    print(f"Learnable parameters: UNet={unet_params:,}  ANNTokenizer={tok_params:,}  Total={unet_params + tok_params:,}", flush=True)

    # Combine UNet + tokenizer parameters for optimizer
    all_params = list(model.parameters())
    if ann_tokenizer is not None:
        all_params += list(ann_tokenizer.parameters())

    # Create a simple namespace that has .parameters() for set_optimiser compatibility
    class _CombinedParams:
        def __init__(self, params):
            self._params = params
        def parameters(self):
            return iter(self._params)

    optimizer = set_optimiser(args, _CombinedParams(all_params))
    loss_function = set_loss_function(args)
    
    # Gradient accumulation setup (must be before LR scheduler)
    accumulation_steps = getattr(args.train, "gradient_accumulation_steps", 1)
    effective_batch_size = args.train.batch_size * accumulation_steps
    print(f"Gradient accumulation: {accumulation_steps} steps, effective batch size: {effective_batch_size}")
    
    # LR scheduler - total_steps equals args.train.steps since we step once per optimizer update
    lr_scheduler = set_learning_rate_scheduler(optimizer, args, total_steps=args.train.steps)

    # EMA setup — shadow weights for smoother evaluation/generation
    ema_decay = float(getattr(args.train, "ema_decay", 0.0))
    ema = None
    if ema_decay > 0.0:
        ema_warmup = int(getattr(args.train, "ema_warmup_steps", 0))
        ema = EMAModel(all_params, decay=ema_decay, warmup_steps=ema_warmup)
        print(f"EMA enabled: decay={ema_decay}, warmup_steps={ema_warmup}")
    else:
        print("EMA disabled (train.ema_decay not set or 0.0)")

    # Resume from checkpoint if specified
    resume_from = getattr(args.train, "resume_from", None)
    start_step = 0
    best_val_r_score = -float('inf')
    if resume_from is not None and os.path.isfile(resume_from):
        start_step, best_val_r_score = load_checkpoint(
            resume_from, model, optimizer, lr_scheduler,
            ann_tokenizer=ann_tokenizer, ema=ema, device=DEVICE
        )
    elif resume_from is not None:
        print(f"WARNING: resume_from path '{resume_from}' not found, training from scratch.", flush=True)

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

    # best_val_r_score is initialised above (or restored from checkpoint)
    if resume_from is None:
        best_val_r_score = -float('inf')

    # Fast-forward the data iterator when resuming so we don't re-train on the same batches
    if start_step > 0:
        print(f"Resuming: skipping {start_step} data batches...", flush=True)
        for _ in range(start_step):
            next(train_iterator)
        print(f"Resuming training from step {start_step}.", flush=True)

    for step in range(start_step, args.train.steps):
        print(f"Step {step+1}/{args.train.steps} started.", flush=True)

        loss_history = []
        train_step(
            step, model, optimizer, lr_scheduler, train_iterator, loss_function,
            diffusion_process, args, autoencoder=autoencoder, loss_history=loss_history,
            use_amp=use_amp, accumulation_steps=accumulation_steps, crossattn_diag=crossattn_diag,
            attn_weights_diag=attn_weights_diag, ann_tokenizer=ann_tokenizer, ema=ema
        )

        ##### Claude idea: run diagnostics every N steps and log to wandb #####
        #if step % 100 == 0:
         #   metrics = run_all_diagnostics(model, valid_dataloader, args, DEVICE)
          #  log_wandb({f"diag/{k}": v for k, v in metrics.items()}, step=step)
        ######

        # Log loss histogram periodically (e.g., every 100 steps)
        histogram_freq = getattr(args.train, "histogram_freq", 100)
        if step % histogram_freq == 0 and step > 0:
            log_loss_histogram(loss_history, step)
        
        if valid_iterator is not None and step % args.validation.eval_freq == 0:
            print(f"Validation at step {step+1}", flush=True)
            # Switch to eval mode for full precision inference (float32)
            model.eval()

            # Swap in EMA weights for validation (smoother, more stable predictions)
            if ema is not None:
                ema.store(all_params)
                ema.copy_to(all_params)

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
                        ann_tokenizer=ann_tokenizer,
                    )

                    if autoencoder is not None:
                        z = generated_samples.squeeze(1)
                        generated_samples = autoencoder.decoder(z)
                    else:
                        generated_samples = generated_samples.squeeze(1)  # remove channel dim

                    current_val_r_score = visualise_and_save_results(generated_samples, step, args, true_fmri=true_fmri)
                else: # toy, mnist and other data with discrete labels
                    generated_samples = diffusivity.generate_samples(args.validation.batch_size, model, diffusion_process, args, device=DEVICE)
                    # TODO: add other image statistics later
                    print("Generated samples shape:", generated_samples.shape, flush=True)
                    visualise_and_save_results(generated_samples, step, args)

            # Save best model BEFORE restoring non-EMA weights, so the
            # checkpoint contains the EMA weights that produced the r-score.
            if current_val_r_score is not None and current_val_r_score > best_val_r_score:
                best_val_r_score = current_val_r_score
                directory_to_save = f"{args.model.output_folder}/{args.jobid}"
                save_checkpoint(directory_to_save, model, optimizer, lr_scheduler, step, best_val_r_score,
                                ann_tokenizer=ann_tokenizer, ema=ema, suffix="best",
                                fmri_min=getattr(args.data, "fmri_min", None), fmri_max=getattr(args.data, "fmri_max", None))
                print(f"New best model saved at step {step} with r-score: {best_val_r_score:.4f}", flush=True)
                log_wandb({"validation/best_r_score": best_val_r_score, "validation/best_step": step}, step=step)

            # Restore original (non-EMA) weights for continued training
            if ema is not None:
                ema.restore(all_params)

        # Periodic checkpoint saving (in addition to best model saving)
        if step % args.model.save_freq == 0 and step > 0:
            directory_to_save = f"{args.model.output_folder}/{args.jobid}"
            save_checkpoint(directory_to_save, model, optimizer, lr_scheduler, step, best_val_r_score,
                            ann_tokenizer=ann_tokenizer, ema=ema, suffix=f"step_{step}",
                            fmri_min=getattr(args.data, "fmri_min", None), fmri_max=getattr(args.data, "fmri_max", None))

    # save the final model
    directory_to_save = f"{args.model.output_folder}/{args.jobid}"
    save_checkpoint(directory_to_save, model, optimizer, lr_scheduler, args.train.steps - 1, best_val_r_score,
                    ann_tokenizer=ann_tokenizer, ema=ema, suffix="final",
                    fmri_min=getattr(args.data, "fmri_min", None), fmri_max=getattr(args.data, "fmri_max", None))
    
    # Cleanup diagnostics
    if crossattn_diag is not None:
        crossattn_diag.cleanup()
    
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
