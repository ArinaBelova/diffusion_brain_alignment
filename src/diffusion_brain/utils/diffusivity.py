# Structure and some functionality is copied from: https://github.com/GabrielNobis/gfdm/blob/main/diffusion/diffusion_lib.py

import torch
import numpy as np
from scipy.integrate import solve_ivp
import torch.nn as nn
from torch.distributions import MultivariateNormal

# custom libs
from abc import ABC, abstractmethod
import einops
from collections.abc import Callable
from typing import Tuple


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def get_diffusion(
    args,
    device="cpu",
):
    """
    Diffusion dynamics constructor

    Args:
        dynamics: name of the dynamic to udr (choose from ['ve','vp'])
        T: terminal time
        device: accelerator device to compute
    Return:
        the diffusion process
    """
    dynamics = args.diffusion.diffusion_type
    T = args.diffusion.T

    if dynamics == "ve":
        return VESDE(args, T=T, device=device)
    elif dynamics == "vp":
        return VPSDE(args, T=T, device=device)
    else:
        raise ValueError(f"Diffusion dynamics {dynamics} not recognized.")
    
class StandardDiffusion(ABC, nn.Module):

    """Abstract class for standard Brownian diffusion processes"""

    def __init__(self, T=1.0, pd_eps=1e-4, device="cpu"):
        super(StandardDiffusion, self).__init__()

        self.register_buffer("T", torch.as_tensor([T], device=device))
        self.pd_eps = pd_eps
        self.device = device 

    def mean_scale(self, t):
        return torch.exp(self.integral(t))
    
    def brown_moments(self, x0, t):
        # print("x0 shape ", x0.shape)
        # print("self.mean_scale(t) shape ", self.mean_scale(t).shape)
        dim_diff = len(x0.shape) - len(t.shape)
        return (
            self.mean_scale(t)[(...,) + (None,) * dim_diff] * x0,
            torch.sqrt(self.var(t)[(...,) + (None,) * dim_diff]),
        )
    
    @abstractmethod
    def f(self, x, t):
        pass 

    @abstractmethod
    def g(self, x, t):
        pass 

    @abstractmethod
    def integral(self, t):
        pass 

    @abstractmethod
    def var(self, t):
        pass

@torch.inference_mode()
def run_forward_sde(process: StandardDiffusion, 
            x_0: torch.Tensor,
            t_0: float = 0.0, 
            T: float = 1.0, 
            n_steps: int = 100,
            **kwargs):
    """Function to run stochastic differential equation. We assume a deterministic initial distribution p_0."""
    
    #Number of trajectories, dimension of data:
    
    n_traj, dim_x = x_0.shape[0], x_0.shape[1:]
    #print("n_traj, dim_x:", n_traj, dim_x)

    #Compute time grid for discretization and step size:
    time_grid = torch.linspace(t_0, T, n_steps)
    dt = time_grid[1] - time_grid[0]
    
    #Initialize list of trajectory:
    x_traj = [x_0]
    #print("time grid ", time_grid)

    for idx, t in enumerate(time_grid):
        #Get last location and time
        x = x_traj[idx]
        t = float(time_grid[idx])
        
        #Get deterministic drift and random drift sample
        determ_drift = process.f(x, t) * dt #= score 

        z = torch.randn(size = (n_traj, *dim_x))
        diffusivity_sample = process.g(x, t) * torch.sqrt(dt) * z #diffusivity_grid_sample[idx]
        
        #Compute next step:
        next_step = x + determ_drift + diffusivity_sample
        
        #Save step:
        x_traj.append(next_step)

    return torch.stack(x_traj, dim = 0), time_grid 

@torch.inference_mode()
def run_reverse_sde(diffusion_process: StandardDiffusion,
            x_0: torch.Tensor,
            score_fn: Callable,
            T: float = 1.0,
            n_steps: int = 1000,
            epsilon=1e-3,
            guidance_scale: float = 1.0,
            label: int = 1,
            num_classes: int = 10,
            cond: torch.Tensor = None,
            device="cpu",
            args=None,
            ann_tokenizer=None,
            identity_label: torch.Tensor = None,
            **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to run reverse-time stochastic differential equation. We assume a deterministic initial Gaussian distribution p_T."""
    score_fn.eval().to(device)

    f = diffusion_process.f
    g = diffusion_process.g

    n_traj, dim_x = x_0.shape[0], x_0.shape[1:]
    # don't start with t = 0 to avoid division by zero in score function
    time_grid = torch.linspace(T, epsilon, n_steps + 1).to(device)
    dt = time_grid[1] - time_grid[0]
    x_traj = [x_0]
    # I preliminary removed all the .long() casting to avoid CUDA crash
        
    # Dual-guidance mode: independent CFG scales for ANN content vs subject identity.
    # When enabled, the additive branch below does 3 forward passes (uncond,
    # ann-only, full) instead of 2 (uncond, full) and composes them as:
    #   score = s_uncond + w_ann*(s_ann - s_uncond) + w_id*(s_full - s_ann)
    # When w_ann == w_id, this reduces to the standard single-guidance formula,
    # so legacy single-scale jobs are untouched. Only active for the additive
    # branch (unet-diffusers with condition_mode="additive") and only when
    # identity_label is provided.
    dual_guidance_flag = bool(getattr(args.validation, "dual_guidance", False)) if args is not None else False
    guidance_scale_identity = float(
        getattr(args.validation, "guidance_scale_identity", guidance_scale)
    ) if args is not None else guidance_scale

    for idx, t in enumerate(time_grid):
        x = x_traj[idx]
        t = torch.tensor([t]).to(device)
        determ_drift = f(x, t)
        z = torch.randn(n_traj, *dim_x).to(device)
        if idx != len(time_grid) - 1:
            diffusivity_sample = g(x, t) * torch.sqrt(torch.abs(dt)) * z
        else:
            diffusivity_sample = 0.0

        # Reset per-iteration. The additive branch sets this when running the
        # 3-pass dual-guidance CFG decomposition; the final composition step
        # checks it to pick the single-vs-dual guidance formula.
        score_ann_only = None

        # TODO: do we really need this clause? It's already covered by below guidance_scale logic
        # if guidance_scale == 1.0:
        #     if args.model.name == "unet-diffusers" or args.model.name == "unet-diffusers-1d":
        #         encoder_hidden_states = torch.zeros(x.shape[0], 1, args.model.cross_attention_dim, device=x.device)
        #         score = score_fn(x, t, encoder_hidden_states = encoder_hidden_states, class_labels = y_target).sample / torch.sqrt(diffusion_process.var(t))
        #     elif args.model.name == "gfdm-unet-1d-cond":
        #         if cond is None:
        #             cond = torch.zeros(x.shape[0], args.model.cross_attention_dim, device=x.device)
        #         score = score_fn(x, t, cond) / torch.sqrt(diffusion_process.var(t))
        #     else:
        #         score = score_fn(x, t, y_target) / torch.sqrt(diffusion_process.var(t))
        # else:
        # for mnist and ann-brain with UNet2DConditionModel:

        debug_conditioning = bool(getattr(args.validation, "debug_conditioning", True)) if args is not None else False

        if args.model.name == "unet-diffusers" or args.model.name == "unet-diffusers-1d":

            #######################
            time_unet = t * 999 # was no casting
            #######################
            
            cond_token_mode = getattr(args.model, "cond_token_mode", "learned")

            def _to_cond_tokens(c):
                """Tokenize ANN vector for cross-attention (mirrors train.py logic)."""
                if ann_tokenizer is not None:
                    return ann_tokenizer(c.float())
                cond_seq_len = max(1, int(getattr(args.model, "cond_seq_len", 1)))
                if cond_token_mode == "chunk" and cond_seq_len > 1 and c.shape[1] % cond_seq_len == 0:
                    return c.reshape(c.shape[0], cond_seq_len, -1).float()
                return c.unsqueeze(1).expand(-1, cond_seq_len, -1).float()

            condition_mode = getattr(args.model, "condition_mode", "cross_attention")

            if condition_mode == "additive" and cond is not None:
                # Pure additive CFG: no cross-attention blocks, ANN through ann_embedding
                # Unconditional: both ANN and identity are None (dropped)
                score_uncond = score_fn(x, time_unet, encoder_hidden_states=None, ann_signal=None, identity_label=None).sample
                # Conditional: pass real ANN signal and identity label
                score_cond = score_fn(x, time_unet, encoder_hidden_states=None, ann_signal=cond.float(), identity_label=identity_label).sample

                # Dual-guidance: extra pass with ANN only (no identity). Used to
                # decompose the CFG direction into "content" (w_ann) and
                # "identity" (w_id) terms. Only meaningful when we actually
                # have a real identity_label to drop in the middle pass.
                if dual_guidance_flag and identity_label is not None:
                    score_ann_only = score_fn(
                        x, time_unet,
                        encoder_hidden_states=None,
                        ann_signal=cond.float(),
                        identity_label=None,
                    ).sample

                if debug_conditioning and idx == 0:
                    delta = (score_cond - score_uncond).abs().mean().item()
                    rel_delta = delta / (score_cond.abs().mean().item() + 1e-8)
                    if score_ann_only is not None:
                        delta_id = (score_cond - score_ann_only).abs().mean().item()
                        print(
                            f"[dual-guidance] unet2d-additive step0: "
                            f"Δ(full-uncond)={delta:.3e}, rel={rel_delta:.3e}, "
                            f"Δ(full-ann_only)={delta_id:.3e} | "
                            f"w_ann={guidance_scale}, w_id={guidance_scale_identity}",
                            flush=True,
                        )
                    elif cond.shape[0] > 1:
                        perm = torch.randperm(cond.shape[0], device=cond.device)
                        score_shuf = score_fn(x, time_unet, encoder_hidden_states=None, ann_signal=cond[perm].float(), identity_label=identity_label).sample
                        delta_shuf = (score_cond - score_shuf).abs().mean().item()
                        print(
                            f"[conditioning-check] unet2d-additive step0: Δ(cond-uncond)={delta:.3e}, "
                            f"rel={rel_delta:.3e}, Δ(cond-shuffled)={delta_shuf:.3e}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[conditioning-check] unet2d-additive step0: Δ(cond-uncond)={delta:.3e}, rel={rel_delta:.3e}",
                            flush=True,
                        )

            elif condition_mode == "cross_attention" and cond is not None:
                # Cross-attention CFG: ANN conditioning through encoder_hidden_states
                encoder_hidden_states_cond = _to_cond_tokens(cond)
                encoder_hidden_states_uncond = torch.zeros_like(encoder_hidden_states_cond)

                score_uncond = score_fn(x, time_unet, encoder_hidden_states=encoder_hidden_states_uncond, class_labels=None).sample
                score_cond = score_fn(x, time_unet, encoder_hidden_states=encoder_hidden_states_cond, class_labels=None).sample

                if debug_conditioning and idx == 0:
                    delta = (score_cond - score_uncond).abs().mean().item()
                    rel_delta = delta / (score_cond.abs().mean().item() + 1e-8)

                    if cond.shape[0] > 1:
                        perm = torch.randperm(cond.shape[0], device=cond.device)
                        encoder_hidden_states_shuf = _to_cond_tokens(cond[perm])
                        score_shuf = score_fn(x, time_unet, encoder_hidden_states=encoder_hidden_states_shuf, class_labels=None).sample
                        delta_shuf = (score_cond - score_shuf).abs().mean().item()
                        print(
                            f"[conditioning-check] unet2d step0: Δ(cond-uncond)={delta:.3e}, "
                            f"rel={rel_delta:.3e}, Δ(cond-shuffled)={delta_shuf:.3e}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[conditioning-check] unet2d step0: Δ(cond-uncond)={delta:.3e}, rel={rel_delta:.3e}",
                            flush=True,
                        )
            else:
                y_target = torch.tensor([label]).repeat(n_traj).to(device) # here was weird repeat(n_traj, 1), did I really need this extra dimension anywhere? #(label + np.zeros((n_traj, 1))).long()
                y_empty = torch.tensor([num_classes]).repeat(n_traj).to(device) 
                # For mnist: use class_labels (discrete), cross-attention gets zeros
                encoder_hidden_states = torch.zeros(x.shape[0], 1, args.model.cross_attention_dim, device=x.device)
                score_uncond = score_fn(x, time_unet, encoder_hidden_states=encoder_hidden_states, class_labels=y_empty).sample
                score_cond = score_fn(x, time_unet, encoder_hidden_states=encoder_hidden_states, class_labels=y_target).sample
        # for ann-brain case with continuous conditioning vector:
        elif args.model.name == "gfdm-unet-1d-cond" or args.model.name == "dit":
            #######################
            time_unet = t * 999 # was no casting
            #######################

            if cond is None:
                cond = torch.zeros(x.shape[0], args.model.cross_attention_dim, device=x.device)

            _cond_mode_1d = getattr(args.model, "condition_mode", "additive")

            if args.model.name == "gfdm-unet-1d-cond" and _cond_mode_1d == "cross_attention":
                # Cross-attention conditioning: tokenize ANN → encoder_out
                def _to_encoder_out(c):
                    tokens = ann_tokenizer(c.float())  # (B, num_tokens, token_dim)
                    return tokens.permute(0, 2, 1).contiguous()  # (B, token_dim, num_tokens)

                encoder_out_cond = _to_encoder_out(cond)
                encoder_out_uncond = torch.zeros_like(encoder_out_cond)
                score_uncond = score_fn(x, time_unet, encoder_out=encoder_out_uncond)
                score_cond = score_fn(x, time_unet, encoder_out=encoder_out_cond)
            elif args.model.name == "gfdm-unet-1d-cond":
                # Additive conditioning: pass raw ANN vector
                cond_uncond = torch.zeros_like(cond).to(device)
                score_uncond = score_fn(x, time_unet, cond=cond_uncond)
                score_cond = score_fn(x, time_unet, cond=cond)
            elif args.model.name == "dit":
                cond_uncond = torch.zeros_like(cond).to(device)
                score_uncond = score_fn(x, time_unet, cond_uncond)
                score_cond = score_fn(x, time_unet, cond)
            else:
                cond_uncond = torch.zeros_like(cond).to(device)
                score_uncond = score_fn(x, time_unet, encoder_out=cond_uncond)
                score_cond = score_fn(x, time_unet, encoder_out=cond)

            if debug_conditioning and idx == 0:
                delta = (score_cond - score_uncond).abs().mean().item()
                rel_delta = delta / (score_cond.abs().mean().item() + 1e-8)

                if cond.shape[0] > 1:
                    perm = torch.randperm(cond.shape[0], device=cond.device)
                    if args.model.name == "gfdm-unet-1d-cond" and _cond_mode_1d == "cross_attention":
                        score_shuf = score_fn(x, time_unet, encoder_out=_to_encoder_out(cond[perm]))
                    elif args.model.name == "gfdm-unet-1d-cond":
                        score_shuf = score_fn(x, time_unet, cond=cond[perm])
                    else:
                        score_shuf = score_fn(x, time_unet, cond[perm])
                    delta_shuf = (score_cond - score_shuf).abs().mean().item()
                    print(
                        f"[conditioning-check] {args.model.name} step0: Δ(cond-uncond)={delta:.3e}, "
                        f"rel={rel_delta:.3e}, Δ(cond-shuffled)={delta_shuf:.3e}",
                        flush=True,
                    )
                else:
                    print(
                        f"[conditioning-check] {args.model.name} step0: Δ(cond-uncond)={delta:.3e}, rel={rel_delta:.3e}",
                        flush=True,
                    )
        else:
            y_target = torch.tensor([label]).repeat(n_traj).to(device) # here was weird repeat(n_traj, 1), did I really need this extra dimension anywhere? #(label + np.zeros((n_traj, 1))).long()
            y_empty = torch.tensor([num_classes]).repeat(n_traj).to(device) 
            print("shape of x and t are: ", x.shape, t.shape, flush=True)
            score_uncond = score_fn(x, t, y_empty)
            score_cond = score_fn(x, t, y_target)
        
        if score_ann_only is not None:
            # Dual-guidance CFG (Imagen / eDiff-I style):
            #   score = s_uncond + w_ann*(s_ann - s_uncond) + w_id*(s_full - s_ann)
            # Independent scales for ANN content vs subject identity.
            score = (
                score_uncond
                + guidance_scale * (score_ann_only - score_uncond)
                + guidance_scale_identity * (score_cond - score_ann_only)
            ) / torch.sqrt(diffusion_process.var(t))
        else:
            score = ((1 - guidance_scale) * score_uncond + guidance_scale * score_cond) / torch.sqrt(diffusion_process.var(t))

        
        ############### DEBUG: Check if scores differ by label#################################
        if debug_conditioning:  # Only at first timestep
            diff_norm = (score_cond - score_uncond).norm()
            print(f"t={time_unet.item():.3f} | " #label={y_target[0].item()} | "
                f"score_cond norm={score_cond.norm():.4f} | "
                f"score_uncond norm={score_uncond.norm():.4f} | "
                f"difference norm={diff_norm:.4f}")
        #################################################################

        # print("x shape ", x.shape)
        # print("t shape ", t.shape)
        # print("score shape ", score.shape)
        # print("determ_drift shape ", determ_drift.shape)
        # print("diffusivity_sample shape ", diffusivity_sample.shape)
        # print("g(x, t)**2  shape ", (g(x, t)**2).shape)

        if args.validation.ode:
            next_step = x + (determ_drift - 0.5 * g(x, t)**2 * score) * dt
        else:
            next_step = x + (determ_drift - g(x, t)**2 * score) * dt + diffusivity_sample

        # --- Per-step Imagen-style thresholding (Saharia et al., 2022) ---
        # With high CFG guidance scales, scores are amplified and push x_t to
        # extreme values mid-trajectory.  Clamping only at the final step cannot
        # recover from a diverged trajectory.  Dynamic thresholding (percentile-
        # based) is safe at every step because the clamp bound adapts to x_t's
        # actual magnitude.  Static thresholding uses the fixed training-data
        # range and is applied only in the second half of the trajectory
        # (t < T/2) where x_t should be approaching the data manifold.
        # thresholding = getattr(args.validation, "thresholding", "none") if args is not None else "none"
        # if thresholding == "dynamic":
        #     p = float(getattr(args.validation, "dynamic_thresholding_percentile", 0.995))
        #     flat = next_step.reshape(next_step.shape[0], -1).abs()
        #     s = torch.quantile(flat, p, dim=1)
        #     s = s.clamp(min=1.0)
        #     for _ in range(len(next_step.shape) - 1):
        #         s = s.unsqueeze(-1)
        #     next_step = next_step.clamp(-s, s)
        # elif thresholding == "static":
        #     # Only clamp in the second half (t < T/2) to avoid interfering
        #     # with legitimately noisy early steps
        #     if t.item() < T / 2:
        #         fmri_min = getattr(args.data, "fmri_min", None)
        #         fmri_max = getattr(args.data, "fmri_max", None)
        #         if fmri_min is not None and fmri_max is not None:
        #             next_step = next_step.clamp(fmri_min, fmri_max)

        x_traj.append(next_step)

    def scale_to_range(tensor, new_min=0.0, new_max=1.0):
        t_min = tensor.min()
        t_max = tensor.max()
        return (tensor - t_min) / (t_max - t_min) * (new_max - new_min) + new_min

    thresholding = getattr(args.validation, "thresholding", "none") if args is not None else "none"
    if thresholding != "none":
        # Final clamp to data range for static thresholding (Saharia et al., 2022)
        fmri_min = getattr(args.data, "fmri_min", None)
        fmri_max = getattr(args.data, "fmri_max", None)
        if fmri_min is not None and fmri_max is not None:
            if thresholding == "static":
                print("Clamping final output with static thresholding to data range [{}, {}]".format(fmri_min, fmri_max), flush=True)
                x_traj[-1] = x_traj[-1].clamp(fmri_min, fmri_max)
            elif thresholding == "scaling":
                print("Scaling final output to data range [{}, {}]".format(fmri_min, fmri_max), flush=True)
                x_traj[-1] = scale_to_range(x_traj[-1], new_min=fmri_min, new_max=fmri_max)

    return x_traj[-1]

@torch.inference_mode()
def generate_samples(num_samples: int,
                     model: nn.Module,
                     diffusion_process: StandardDiffusion,
                     args,
                     device,
                     cond: torch.Tensor = None,
                     ann_tokenizer=None,
                     identity_label: torch.Tensor = None):
    """Function to generate samples from the learned diffusion model"""
    # initial samples from p_T
    raw_model = _unwrap_model(model)
    
    # Track original size for cropping 2D data back
    original_size = getattr(args.model, "input_size_original", None)
    
    if args.data.data_name == "toy":
        dim_x = [args.model.input_size]
    elif args.model.name == "gfdm-unet-1d-cond":
        orig_len = args.model.input_size[0]
        factor = getattr(raw_model, "downsample_factor", 1)
        if factor > 1 and orig_len % factor != 0:
            pad_len = (factor - (orig_len % factor)) % factor
        else:
            pad_len = 0
        padded_len = orig_len + pad_len
        dim_x = (args.model.c_in, padded_len)
    elif args.model.name == "dit":
        dim_x = [args.model.input_size]
    elif args.data.data_name == "mnist": 
        dim_x = (args.model.c_in, args.model.input_size, args.model.input_size)
    elif args.model.name == "unet-diffusers" or args.model.name == "unet-diffusers-1d":
        # For 2D brain data: pad to the same multiple used during training
        # so the UNet sees the same spatial dimensions it was trained on.
        # (mirrors compute_2d_padding / pad_2d_to_multiple from train.py)
        c, h, w = args.model.input_size
        multiple = int(getattr(args.data, "resize_to_multiple", 8))
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        dim_x = (c, h + pad_h, w + pad_w)
        original_size = (c, h, w)  # override for cropping back
    else:
        raise ValueError(f"Model {args.model.name} not recognized for sample generation.")    

    print("In generation generating with model {} and diffusion process {}, noise shape will be {}".format(args.model.name, diffusion_process, (num_samples, *dim_x)), flush=True)
    noise = torch.randn(size=(num_samples, *dim_x), device=device)
    mu, std = diffusion_process.brown_moments(torch.zeros(num_samples, *dim_x).to(device), diffusion_process.T)


    # print("in generate_samples noise shape is {}".format(noise.shape), flush=True)
    # print("in generate_samples mu and std shapes:", mu.shape, std.shape, flush=True)
    
    x_T = std * noise # + mu # mean was commented out 
    # was _, x_0 as we had also trajectory tracked, but no need for that for the sake of generation speed
    x_0 = run_reverse_sde(
        diffusion_process=diffusion_process,
        x_0=x_T, # .cpu().numpy()
        score_fn=model,
        T=diffusion_process.T.item(),
        n_steps=args.validation.n_steps,
        guidance_scale=args.validation.guidance_scale,
        label=args.validation.label_to_generate,
        num_classes=args.model.num_classes,
        cond=cond,
        score_scaling=True,
        device=device,
        args=args,
        ann_tokenizer=ann_tokenizer,
        identity_label=identity_label,
    )

    if args.model.name == "gfdm-unet-1d-cond":
        x_0 = x_0[..., :orig_len]
    
    # Crop 2D brain data back to original size
    is_2d = getattr(args.data, "is_2d", False)
    if is_2d and original_size is not None and len(original_size) == 3:
        _, orig_h, orig_w = original_size
        x_0 = x_0[..., :orig_h, :orig_w]

    # print("x_0 and label shapes are: ", x_0.shape, cond.shape, flush=True)    
    return x_0 # * 255 as I don't really know what scale the model learned...

class VESDE(StandardDiffusion):

    """Variance exploding standard Brownian diffusion process"""

    def __init__(
            self,
            args,
            T = 1.0,
            device="cpu"
    ):
            
        super().__init__(T=T, device=device)

        self.name = "ve"

        self.register_buffer(
            "sigma_min", torch.as_tensor(torch.tensor([args.diffusion.sigma_min]), device=self.device)
        )

        self.register_buffer(
            "sigma_max", torch.as_tensor(torch.tensor([args.diffusion.sigma_max]), device=self.device)
        )

        self.register_buffer(
            "r", torch.as_tensor(self.sigma_max / self.sigma_min, device=self.device)
        )

        self.register_buffer(
            "a",
            torch.as_tensor(
                self.sigma_min * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min))),
                device=self.device,
            ),
        )

    def f(self, x, t):
        return 0 * torch.ones_like(t)

    def g(self, x, t):
        return self.a * (self.r ** t)

    def integral(self, t):
        return 0 * t 

    def var(self, t):
        return (self.sigma_min ** 2) * ((self.sigma_max / self.sigma_min) ** (2 * t)) 


class VPSDE(StandardDiffusion):

    """Variance preseving standard Brownian diffusion process"""
    
    def __init__(
            self,
            args,
            T = 1.0,
            device="cpu"
    ):
            
        super().__init__(T=T, device=device)

        self.name = "vp"
        self.register_buffer(
            "beta_min", torch.as_tensor(torch.tensor([args.diffusion.beta_min]), device=self.device)
        )
        self.register_buffer(
            "beta_max", torch.as_tensor(torch.tensor([args.diffusion.beta_max]), device=self.device)
        )

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def mu(self, t):
        return -0.5 * self.beta(t)
    
    def f(self, x, t):
        return self.mu(t) * x

    def g(self, x, t):
        return torch.sqrt(self.beta(t))

    # TODO check this!!!
    def integral(self, t):
        return -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def var(self, t):
        scale = -0.5 * (t ** 2) * (self.beta_max - self.beta_min) - t * self.beta_min
        return 1 - torch.exp(scale)
        #return 1 - torch.exp(-self.beta(t))              
