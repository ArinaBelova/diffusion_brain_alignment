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
    dynamics="ve",
    T=1.0,
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

    constructors = {
        "ve": VESDE,
        "vp": VPSDE,
    }

    return constructors[dynamics.lower()](
        T=T,
        device=device,
    )

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
        return (
            self.mean_scale(t)[:, None, None, None] * x0,
            torch.sqrt(self.var(t))[:, None, None, None],
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
    print("n_traj, dim_x:", n_traj, dim_x)

    #Compute time grid for discretization and step size:
    time_grid = torch.linspace(t_0, T, n_steps)
    dt = time_grid[1] - time_grid[0]
    
    #Initialize list of trajectory:
    x_traj = [x_0]
    print("time grid ", time_grid)
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

    return torch.stack(x_traj, dim = 0), time_grid #torch.stack(x_traj, dim = 0), time_grid   

@torch.inference_mode()
def run_reverse_sde(diffusion_process: StandardDiffusion,
            x_0: np.ndarray,
            score_fn: Callable,
            # forward_x_0: np.ndarray,
            # t_0: float = 0.0, 
            T: float = 1.0, 
            n_steps: int = 1000,
            # injected_noises = None,
            epsilon=1e-3,
            guidance_scale: float = 1.0,
            label: int = 0,
            num_classes: int = 10,
            device="cpu",
            **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to run reverse-time stochastic differential equation. We assume a deterministic initial Gaussian distribution p_T."""
    score_fn.eval()

    f = diffusion_process.f
    g = diffusion_process.g

    n_traj, dim_x = x_0.shape[0], x_0.shape[1:]
    # don't start with t = 0 to avoid division by zero in score function
    time_grid = np.linspace(T, epsilon, n_steps + 1, device=device)
    dt = time_grid[1] - time_grid[0]

    # When we thought the precision was the problem:
    # reverse_x_0 = reverse_x_0.astype(np.float64)
    # forward_x_0 = forward_x_0.astype(np.float64)
    # injected_noises = injected_noises.astype(np.float64)
    # time_grid = time_grid.astype(np.float64)

    x_traj = [x_0]
    y_target = (label + np.zeros(n_traj, 1)).long()
    y_empty = num_classes + np.zeros(n_traj, 1).long()

    for idx, t in enumerate(time_grid):
        x = x_traj[idx]
        t = time_grid[idx]

        determ_drift = f(x, t) * dt

        z = np.random.randn(n_traj, *dim_x) 
        diffusivity_sample = g(x, t) * np.sqrt(np.abs(dt)) * z 

        if guidance_scale == 1.0:
            score = score_fn(x = x, t = t, y = y_target) / np.sqrt(diffusion_process.var(t))
        else:
            score_uncond = score_fn(x = x, t = t, y = y_empty)
            score_cond = score_fn(x = x, t = t, y = y_target)
            score = ((1 - guidance_scale) * score_uncond + guidance_scale * score_cond) / np.sqrt(diffusion_process.var(t))

        next_step = x + determ_drift - g(x, t)**2 * score * dt + diffusivity_sample
        #print(f"time {t} next_step: ", next_step) # (1000, 1000)

        x_traj.append(next_step)
    
    return np.stack(x_traj), next_step

@torch.inference_mode()
def generate_samples(num_samples: int,
                     model: nn.Module,
                     diffusion_process: StandardDiffusion,
                     args):
    """Function to generate samples from the learned diffusion model"""
    # initial samples from p_T
    dim_x = (args.data.channels, args.data.input_size, args.data.input_size)
    x_T = torch.randn(size=(num_samples, *dim_x), device=args.device)
    
    _, x_0 = run_reverse_sde(
        diffusion_process=diffusion_process,
        x_0=x_T.cpu().numpy(),
        score_fn=model,
        T=diffusion_process.T.item(),
        n_steps=args.sampling.n_steps,
        guidance_scale=args.sampling.guidance_scale,
        label=args.sampling.label_to_generate,
        num_classes=args.model.num_classes,
        score_scaling=True,
        device=args.device,
    )
    return x_0

class VESDE(StandardDiffusion):

    """Variance exploding standard Brownian diffusion process"""

    def __init__(
            self,
            sigma_min = 0.01,
            sigma_max = 50.0,
            T = 1.0,
            device="cpu"
    ):
            
        super().__init__(T=T, device=device)

        self.name = "ve"

        self.register_buffer(
            "sigma_min", torch.as_tensor(torch.tensor([sigma_min]), device=self.device)
        )

        self.register_buffer(
            "sigma_max", torch.as_tensor(torch.tensor([sigma_max]), device=self.device)
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
            beta_min = 0.1, #0.01, # as in Gabriel's work
            beta_max = 20, # 50.0,
            T = 1.0,
            device="cpu"
    ):
            
        super().__init__(T=T, device=device)

        self.name = "vp"
        self.register_buffer(
            "beta_min", torch.as_tensor(torch.tensor([beta_min]), device=self.device)
        )
        self.register_buffer(
            "beta_max", torch.as_tensor(torch.tensor([beta_max]), device=self.device)
        )

    # TODO: decide if we use this beta_t or calculate beta in mu.g ourselves
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def mu(self, t):
        return -0.5 * self.beta(t) #(self.beta_max - self.beta_min) * torch.ones_like(t) 
    
    def f(self, x, t):
        return self.mu(t) * x

    def g(self, x, t):
        return torch.sqrt(self.beta(t)) #torch.sqrt(self.beta_max - self.beta_min) * torch.ones_like(t)

    # TODO check this!!!
    def integral(self, t):
        return -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        #return -0.5 * torch.cumsum(self.beta_t(t), dim=0) * (t[1] - t[0]) 

    def var(self, t):
        return 1 - torch.exp(-self.beta(t))              