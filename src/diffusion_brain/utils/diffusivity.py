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
    y_target = torch.tensor([label]).repeat(n_traj).to(device) # here was weird repeat(n_traj, 1), did I really need this extra dimension anywhere? #(label + np.zeros((n_traj, 1))).long()
    y_empty = torch.tensor([num_classes]).repeat(n_traj).to(device) 
        
    for idx, t in enumerate(time_grid):
        x = x_traj[idx]
        t = torch.tensor([t]).to(device)
        determ_drift = f(x, t)
        z = torch.randn(n_traj, *dim_x).to(device) 
        if idx != len(time_grid) - 1:
            diffusivity_sample = g(x, t) * torch.sqrt(torch.abs(dt)) * z 
        else:
            diffusivity_sample = 0.0 

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
        if args.model.name == "unet-diffusers" or args.model.name == "unet-diffusers-1d":

            #######################
            time_unet = t * 999
            #######################
            
            encoder_hidden_states = torch.zeros(x.shape[0], 1, args.model.cross_attention_dim, device=x.device)

            # print("y_empty:", y_empty.device, y_empty.shape, y_empty.dtype)
            # print("y_target:", y_target.device, y_target.shape, y_target.dtype)
            # print("x: ", x.device, x.shape, x.dtype)
            # print("t: ", t.device, t.shape, t.dtype)
            # print("encoder_hidden_states: ", encoder_hidden_states.device, encoder_hidden_states.shape, encoder_hidden_states.dtype)

            score_uncond = score_fn(x, time_unet, encoder_hidden_states = encoder_hidden_states, class_labels=y_empty).sample
            score_cond = score_fn(x, time_unet, encoder_hidden_states = encoder_hidden_states, class_labels=y_target).sample
        elif args.model.name == "gfdm-unet-1d-cond":
            if cond is None:
                cond = torch.zeros(x.shape[0], args.model.cross_attention_dim, device=x.device)
            cond_uncond = torch.zeros_like(cond).to(device)
            score_uncond = score_fn(x, t, cond_uncond)
            score_cond = score_fn(x, t, cond)
        else:
            score_uncond = score_fn(x, t, y_empty)
            score_cond = score_fn(x, t, y_target)
        
        score = ((1 - guidance_scale) * score_uncond + guidance_scale * score_cond) / torch.sqrt(diffusion_process.var(t))

        # DEBUG: Check if scores differ by label#################################
        # if idx == 0:  # Only at first timestep
        #     diff_norm = (score_cond - score_uncond).norm()
        #     print(f"t={t.item():.3f} | label={y_target[0].item()} | "
        #         f"score_cond norm={score_cond.norm():.4f} | "
        #         f"score_uncond norm={score_uncond.norm():.4f} | "
        #         f"difference norm={diff_norm:.4f}")
        ##################################################################
            
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

        x_traj.append(next_step)
    
    #return torch.stack(x_traj), next_step
    return next_step 

@torch.inference_mode()
def generate_samples(num_samples: int,
                     model: nn.Module,
                     diffusion_process: StandardDiffusion,
                     args,
                     device,
                     cond: torch.Tensor = None):
    """Function to generate samples from the learned diffusion model"""
    # initial samples from p_T
    if args.data.data_name == "toy":
        dim_x = [args.model.input_size]
    elif args.model.name == "gfdm-unet-1d-cond":
        orig_len = args.model.input_size
        factor = getattr(model, "downsample_factor", 1)
        if factor > 1 and orig_len % factor != 0:
            pad_len = (factor - (orig_len % factor)) % factor
        else:
            pad_len = 0
        padded_len = orig_len + pad_len
        dim_x = (args.model.c_in, padded_len)
        args.validation.label_to_generate = 1 # in reality we don't use it, it's just a stab
    else:
        dim_x = (args.model.c_in, args.model.input_size, args.model.input_size)

    noise = torch.randn(size=(num_samples, *dim_x), device=device)
    _, std = diffusion_process.brown_moments(torch.zeros(num_samples, *dim_x).to(device), diffusion_process.T)

    # print("in generate_samples mu and std shapes:", mu.shape, std.shape, flush=True)
    
    x_T = std * noise # + mu
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
        args=args
    )

    if args.model.name == "gfdm-unet-1d-cond":
        x_0 = x_0[..., :orig_len]

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
