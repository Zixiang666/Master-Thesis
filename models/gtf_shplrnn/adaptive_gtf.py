#!/usr/bin/env python3
"""
Adaptive Generalized Teacher Forcing (GTF) Implementation
=========================================================

Ported from TRR_WS project with enhancements for ECG classification.
Provides adaptive alpha estimation based on model's spectral radius
for improved training stability and convergence.

Author: Master Thesis Project
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gmean
from typing import Optional, Literal, Dict, Tuple
from dataclasses import dataclass


@dataclass
class AdaptiveGTFConfig:
    """Configuration for Adaptive GTF"""
    initial_alpha: float = 0.1
    gamma: float = 0.95  # Smoothing factor for alpha updates
    update_step: int = 10  # Update alpha every N optimization steps
    estimation_method: Literal["geometric_mean", "arithmetic_mean", "upper_bound"] = "geometric_mean"
    alpha_min: float = 0.01
    alpha_max: float = 0.5
    adaptive: bool = True


class AdaptiveGTF:
    """
    Adaptive Generalized Teacher Forcing with spectral norm-based alpha estimation.
    
    The alpha parameter controls the mixing between teacher forcing (encoded states)
    and free running (predicted states) during training. This implementation
    automatically adjusts alpha based on the model's spectral radius to ensure
    stable training dynamics.
    
    Reference: "Generalized Teacher Forcing for Learning Chaotic Dynamics"
    """
    
    def __init__(self, config: AdaptiveGTFConfig):
        """
        Initialize Adaptive GTF.
        
        Args:
            config: Configuration object with GTF parameters
        """
        self.config = config
        self.alpha = config.initial_alpha
        self.gamma = config.gamma
        self.update_step = config.update_step
        self.alpha_min = config.alpha_min
        self.alpha_max = config.alpha_max
        self.adaptive = config.adaptive
        
        # Statistics tracking
        self.alpha_history = []
        self.spectral_norm_history = []
        
        # Select estimation method
        if config.estimation_method == "geometric_mean":
            self.compute_spectral_norm = self._geometric_mean
        elif config.estimation_method == "arithmetic_mean":
            self.compute_spectral_norm = self._arithmetic_mean
        elif config.estimation_method == "upper_bound":
            self.compute_spectral_norm = self._upper_bound
        else:
            raise ValueError(f"Unknown estimation method: {config.estimation_method}")
    
    @torch.no_grad()
    def update_alpha(self, 
                    jacobians: torch.Tensor,
                    optimization_step: int) -> float:
        """
        Update alpha based on model's Jacobian spectral norm.
        
        Args:
            jacobians: Jacobian matrices [B, T, N, N] where B=batch, T=time, N=latent_dim
            optimization_step: Current optimization iteration
            
        Returns:
            Updated alpha value
        """
        if not self.adaptive:
            return self.alpha
            
        if optimization_step % self.update_step == 0:
            # Compute spectral norm using selected method
            spectral_norm = self.compute_spectral_norm(jacobians)
            self.spectral_norm_history.append(spectral_norm.item())
            
            # Estimate optimal alpha based on spectral radius
            # Formula: α = max(0, 1 - 1/ρ(J)) ensures stability
            estimated_alpha = max(0, 1.0 - 1.0 / (spectral_norm + 1e-8))
            estimated_alpha = min(estimated_alpha, self.alpha_max)
            
            # Apply smoothing with gamma factor
            if estimated_alpha > self.alpha:
                # Increase alpha quickly if model is stable
                self.alpha = estimated_alpha
            else:
                # Decrease alpha slowly with smoothing
                self.alpha = max(
                    self.gamma * self.alpha + (1.0 - self.gamma) * estimated_alpha,
                    self.alpha_min
                )
            
            self.alpha_history.append(self.alpha)
            
        return self.alpha
    
    def _arithmetic_mean(self, jacobians: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral norm using arithmetic mean over time.
        
        Args:
            jacobians: [B, T, N, N] Jacobian matrices
            
        Returns:
            Maximum spectral norm across batch
        """
        # Average Jacobians over time dimension
        mean_jacobians = torch.mean(jacobians, dim=1)  # [B, N, N]
        
        # Compute spectral norm (largest singular value) for each batch
        spectral_norms = torch.linalg.matrix_norm(mean_jacobians, ord=2, dim=(1, 2))
        
        # Return maximum across batch
        return torch.max(spectral_norms)
    
    def _geometric_mean(self, jacobians: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral norm using geometric mean over time.
        More suitable for multiplicative dynamics in RNNs.
        
        Args:
            jacobians: [B, T, N, N] Jacobian matrices
            
        Returns:
            Maximum spectral norm across batch
        """
        # Move to CPU for scipy computation (geometric mean)
        jacobians_cpu = jacobians.cpu().numpy()
        
        # Geometric mean over time dimension using log-sum-exp trick
        # This is more stable for long sequences
        geometric_mean_jacobians = gmean(jacobians_cpu, axis=1)  # [B, N, N]
        
        # Convert back to torch and compute spectral norm
        geometric_mean_jacobians = torch.from_numpy(geometric_mean_jacobians).to(jacobians.device)
        spectral_norms = torch.linalg.matrix_norm(geometric_mean_jacobians, ord=2, dim=(1, 2))
        
        return torch.max(spectral_norms)
    
    def _upper_bound(self, jacobians: torch.Tensor) -> torch.Tensor:
        """
        Compute upper bound of spectral norm (conservative estimate).
        Takes spectral norm first, then geometric mean.
        
        Args:
            jacobians: [B, T, N, N] Jacobian matrices
            
        Returns:
            Maximum spectral norm across batch
        """
        # Compute spectral norm for each time step
        spectral_norms_per_time = torch.linalg.matrix_norm(jacobians, ord=2, dim=(2, 3))  # [B, T]
        
        # Geometric mean of spectral norms over time
        spectral_norms_cpu = spectral_norms_per_time.cpu().numpy()
        geometric_mean_norms = gmean(spectral_norms_cpu, axis=1)  # [B]
        
        # Return maximum across batch
        return torch.max(torch.from_numpy(geometric_mean_norms))
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current GTF statistics.
        
        Returns:
            Dictionary with current alpha and recent spectral norms
        """
        stats = {
            "alpha": self.alpha,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
        }
        
        if self.spectral_norm_history:
            stats["spectral_norm_mean"] = np.mean(self.spectral_norm_history[-10:])
            stats["spectral_norm_std"] = np.std(self.spectral_norm_history[-10:])
        
        return stats
    
    def reset(self):
        """Reset alpha to initial value and clear history."""
        self.alpha = self.config.initial_alpha
        self.alpha_history = []
        self.spectral_norm_history = []


class AlphaScheduler:
    """
    Milestone-based alpha scheduler for gradual teacher forcing reduction.
    Allows smooth transition from teacher forcing to free running.
    """
    
    def __init__(self,
                 initial_alpha: float,
                 milestones: list,
                 target_values: list):
        """
        Initialize alpha scheduler.
        
        Args:
            initial_alpha: Starting alpha value
            milestones: Epoch numbers for alpha changes
            target_values: Target alpha values at each milestone
        """
        assert len(milestones) == len(target_values), "Milestones and targets must have same length"
        
        self.initial_alpha = initial_alpha
        self.milestones = milestones
        self.target_values = target_values
        self.current_alpha = initial_alpha
    
    def get_alpha(self, epoch: int) -> float:
        """
        Get alpha value for current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Scheduled alpha value
        """
        # Find current milestone interval
        for i, milestone in enumerate(self.milestones):
            if epoch < milestone:
                if i == 0:
                    # Before first milestone
                    starting_value = self.initial_alpha
                    last_milestone = 0
                else:
                    # Between milestones
                    starting_value = self.target_values[i - 1]
                    last_milestone = self.milestones[i - 1]
                
                target_value = self.target_values[i]
                
                # Exponential decay between milestones
                decay_rate = (target_value / starting_value) ** (1 / (milestone - last_milestone))
                self.current_alpha = starting_value * (decay_rate ** (epoch - last_milestone))
                break
        else:
            # After last milestone
            self.current_alpha = self.target_values[-1]
        
        return self.current_alpha
    
    def step(self, epoch: int):
        """Update internal state for epoch."""
        self.current_alpha = self.get_alpha(epoch)


class CompositeScheduler:
    """
    Unified scheduler for multiple alpha values and learning rate.
    Manages alpha values for different loss components.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 n_epochs: int,
                 config: Dict):
        """
        Initialize composite scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            n_epochs: Total number of training epochs
            config: Configuration dictionary with scheduler parameters
        """
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.config = config
        self.current_epoch = 0
        
        # Initialize individual schedulers
        self.schedulers = {}
        
        # Alpha schedulers for different components
        alpha_components = ["gtf", "reconstruction", "dsr", "consistency", "entropy", "mar"]
        for component in alpha_components:
            if f"alpha_{component}" in config:
                alpha_config = config[f"alpha_{component}"]
                if isinstance(alpha_config, dict):
                    self.schedulers[f"alpha_{component}"] = AlphaScheduler(
                        alpha_config.get("initial", 0.1),
                        alpha_config.get("milestones", [n_epochs // 2]),
                        alpha_config.get("targets", [0.01])
                    )
                else:
                    # Constant alpha
                    self.schedulers[f"alpha_{component}"] = lambda epoch: alpha_config
        
        # Learning rate scheduler
        if "lr_scheduler" in config:
            lr_config = config["lr_scheduler"]
            if lr_config["type"] == "cosine":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_epochs, eta_min=lr_config.get("min_lr", 1e-6)
                )
            elif lr_config["type"] == "step":
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=lr_config.get("step_size", 30),
                    gamma=lr_config.get("gamma", 0.1)
                )
            else:
                self.lr_scheduler = None
        else:
            self.lr_scheduler = None
    
    def value(self, key: str, largest_possible: bool = False) -> float:
        """
        Get current value for a scheduled parameter.
        
        Args:
            key: Parameter name (e.g., "alpha_gtf")
            largest_possible: If True, return maximum possible value
            
        Returns:
            Current parameter value
        """
        if key not in self.schedulers:
            return self.config.get(key, 0.0)
        
        scheduler = self.schedulers[key]
        if callable(scheduler):
            return scheduler(self.current_epoch)
        elif hasattr(scheduler, 'current_alpha'):
            return scheduler.current_alpha if not largest_possible else scheduler.initial_alpha
        else:
            return self.config.get(key, 0.0)
    
    def is_used(self, key: str) -> bool:
        """
        Check if a parameter is being used (non-zero).
        
        Args:
            key: Parameter name
            
        Returns:
            True if parameter is active
        """
        return self.value(key) > 0
    
    def step(self, metric: Optional[float] = None):
        """
        Step all schedulers forward.
        
        Args:
            metric: Optional metric for ReduceLROnPlateau scheduler
        """
        self.current_epoch += 1
        
        # Step alpha schedulers
        for key, scheduler in self.schedulers.items():
            if hasattr(scheduler, 'step'):
                scheduler.step(self.current_epoch)
        
        # Step learning rate scheduler
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.lr_scheduler.step(metric)
            else:
                self.lr_scheduler.step()
    
    def get_stats(self) -> Dict[str, float]:
        """Get current values of all scheduled parameters."""
        stats = {}
        
        # Alpha values
        for key in self.schedulers:
            stats[key] = self.value(key)
        
        # Learning rate
        if self.optimizer is not None:
            stats["lr"] = self.optimizer.param_groups[0]["lr"]
        
        stats["epoch"] = self.current_epoch
        
        return stats


def compute_jacobian(model: nn.Module,
                     z: torch.Tensor,
                     create_graph: bool = False) -> torch.Tensor:
    """
    Compute Jacobian matrix of model dynamics.
    
    Args:
        model: Neural network model with forward method
        z: Input tensor [B, T, N] where N is latent dimension
        create_graph: Whether to create computation graph for higher-order gradients
        
    Returns:
        Jacobian matrices [B, T-1, N, N]
    """
    B, T, N = z.shape
    
    # We compute Jacobian for transitions z_t -> z_{t+1}
    jacobians = []
    
    for t in range(T - 1):
        z_t = z[:, t, :]  # [B, N]
        z_t.requires_grad_(True)
        
        # Forward pass through model
        z_next = model(z_t.unsqueeze(1)).squeeze(1)  # [B, N]
        
        # Compute Jacobian for each batch element
        batch_jacobians = []
        for b in range(B):
            jacobian_rows = []
            for i in range(N):
                # Compute gradient of output i with respect to all inputs
                grad = torch.autograd.grad(
                    z_next[b, i],
                    z_t,
                    retain_graph=True,
                    create_graph=create_graph
                )[0][b]  # [N]
                jacobian_rows.append(grad)
            
            jacobian = torch.stack(jacobian_rows)  # [N, N]
            batch_jacobians.append(jacobian)
        
        jacobians.append(torch.stack(batch_jacobians))  # [B, N, N]
    
    return torch.stack(jacobians, dim=1)  # [B, T-1, N, N]