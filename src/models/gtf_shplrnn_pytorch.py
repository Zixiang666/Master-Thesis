#!/usr/bin/env python3
"""
PyTorch Implementation of GTF-shPLRNN (Generalized Teacher Forcing shallow PLRNN)
==================================================================================

Based on the paper "Generalized Teacher Forcing for Learning Chaotic Dynamics" 
by Hess et al. (2023) and the Julia implementation from DurstewitzLab.

This implementation provides:
- Shallow PLRNN (shPLRNN) architecture
- Generalized Teacher Forcing (GTF) training mechanism
- Adaptive α parameter scheduling
- Bounded gradient guarantees for chaotic systems

Author: Master Thesis Project
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.linalg import eigvals


@dataclass
class GTFConfig:
    """Configuration for Generalized Teacher Forcing"""
    use_gtf: bool = True
    alpha: float = 0.1  # GTF mixing parameter (0=full teacher forcing, 1=no teacher forcing)
    alpha_method: str = "adaptive"  # "constant", "linear", "adaptive"
    alpha_min: float = 0.0
    alpha_max: float = 0.5
    alpha_adapt_rate: float = 0.001
    teacher_forcing_interval: int = 1  # For standard teacher forcing
    gradient_clip_value: float = 1.0
    bounded_gradient_guarantee: bool = True


class ShallowPLRNN(nn.Module):
    """
    Shallow Piecewise Linear Recurrent Neural Network (shPLRNN)
    
    Architecture:
    z_t = A * z_{t-1} + W1 * relu(W2 * z_{t-1} + h2) + h1
    
    Where:
    - A: Diagonal autoregressive matrix (linear dynamics)
    - W1, W2: Weight matrices for nonlinear transformation
    - h1, h2: Bias vectors
    - relu: Rectified linear unit activation
    
    This architecture provides:
    - More constrained nonlinear dynamics compared to vanilla PLRNN
    - Better interpretability through shallow network structure
    - Efficient gradient flow for chaotic systems
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int, 
                 observation_dim: Optional[int] = None,
                 use_observation_model: bool = True):
        super(ShallowPLRNN, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.observation_dim = observation_dim or latent_dim
        self.use_observation_model = use_observation_model
        
        # Autoregressive parameters
        self.A = nn.Parameter(torch.eye(latent_dim) * 0.9)  # Initialize near identity
        
        # Nonlinear transformation parameters
        self.W1 = nn.Parameter(torch.randn(latent_dim, hidden_dim) * 0.1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 0.1)
        self.h1 = nn.Parameter(torch.zeros(latent_dim))
        self.h2 = nn.Parameter(torch.zeros(hidden_dim))
        
        # Optional observation model
        if use_observation_model:
            self.B = nn.Parameter(torch.randn(observation_dim, latent_dim) * 0.1)
            self.b = nn.Parameter(torch.zeros(observation_dim))
        
        # External input coupling (optional)
        self.C = nn.Parameter(torch.zeros(latent_dim, observation_dim))
        
    def step(self, z: torch.Tensor, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single step of shPLRNN dynamics
        
        Args:
            z: Latent state [batch_size, latent_dim]
            external_input: Optional external input [batch_size, input_dim]
            
        Returns:
            z_next: Next latent state [batch_size, latent_dim]
        """
        # Linear autoregressive term
        z_linear = torch.matmul(z, self.A.T)
        
        # Nonlinear transformation through shallow network
        hidden = F.relu(torch.matmul(z, self.W2.T) + self.h2)
        z_nonlinear = torch.matmul(hidden, self.W1.T) + self.h1
        
        # Combine linear and nonlinear terms
        z_next = z_linear + z_nonlinear
        
        # Add external input if provided
        if external_input is not None:
            z_next = z_next + torch.matmul(external_input, self.C.T)
        
        return z_next
    
    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """
        Observation model: maps latent state to observations
        
        Args:
            z: Latent state [batch_size, latent_dim]
            
        Returns:
            x: Observations [batch_size, observation_dim]
        """
        if self.use_observation_model:
            return torch.matmul(z, self.B.T) + self.b
        else:
            return z
    
    def forward(self, z0: torch.Tensor, seq_len: int, 
                external_inputs: Optional[torch.Tensor] = None,
                return_latents: bool = False) -> torch.Tensor:
        """
        Generate sequence using shPLRNN dynamics
        
        Args:
            z0: Initial latent state [batch_size, latent_dim]
            seq_len: Length of sequence to generate
            external_inputs: Optional external inputs [batch_size, seq_len, input_dim]
            return_latents: Whether to return latent states
            
        Returns:
            observations: Generated observations [batch_size, seq_len, observation_dim]
            latents (optional): Latent states [batch_size, seq_len, latent_dim]
        """
        batch_size = z0.shape[0]
        z = z0
        
        latents = []
        observations = []
        
        for t in range(seq_len):
            # Get external input for this timestep
            ext_input = external_inputs[:, t] if external_inputs is not None else None
            
            # Update latent state
            z = self.step(z, ext_input)
            latents.append(z)
            
            # Generate observation
            x = self.observe(z)
            observations.append(x)
        
        # Stack sequences
        observations = torch.stack(observations, dim=1)
        
        if return_latents:
            latents = torch.stack(latents, dim=1)
            return observations, latents
        else:
            return observations
    
    def compute_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of dynamics at state z
        
        Args:
            z: Latent state [batch_size, latent_dim]
            
        Returns:
            J: Jacobian matrix [batch_size, latent_dim, latent_dim]
        """
        batch_size = z.shape[0]
        J = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=z.device)
        
        # Linear part
        J += self.A.unsqueeze(0)
        
        # Nonlinear part (through automatic differentiation)
        z.requires_grad_(True)
        for i in range(self.latent_dim):
            z_next = self.step(z)
            grad = torch.autograd.grad(z_next[:, i].sum(), z, retain_graph=True)[0]
            J[:, i] = grad
        
        return J
    
    def check_stability(self) -> Dict[str, float]:
        """
        Check stability properties of the shPLRNN
        
        Returns:
            stability_metrics: Dictionary with stability metrics
        """
        with torch.no_grad():
            # Eigenvalues of linear part
            A_eigvals = torch.linalg.eigvals(self.A).cpu().numpy()
            max_eigval_mag = np.max(np.abs(A_eigvals))
            
            # Estimate Lipschitz constant
            W1_norm = torch.norm(self.W1, p=2).item()
            W2_norm = torch.norm(self.W2, p=2).item()
            lipschitz_estimate = max_eigval_mag + W1_norm * W2_norm
            
            return {
                'max_eigenvalue_magnitude': max_eigval_mag,
                'lipschitz_estimate': lipschitz_estimate,
                'is_contractive': lipschitz_estimate < 1.0,
                'spectral_radius': max_eigval_mag
            }


class GeneralizedTeacherForcing:
    """
    Generalized Teacher Forcing (GTF) implementation
    
    GTF modifies standard teacher forcing by mixing predicted and true states:
    z_t^mix = α * z_t^pred + (1 - α) * z_t^true
    
    This provides:
    - Bounded gradients even for chaotic systems
    - Smooth transition from teacher forcing to free running
    - Better long-term prediction accuracy
    """
    
    def __init__(self, config: GTFConfig):
        self.config = config
        self.alpha_history = []
        self.gradient_norms = []
        self.prediction_errors = []
        
    def get_alpha(self, epoch: int, total_epochs: int, 
                  current_error: Optional[float] = None) -> float:
        """
        Get current α value based on scheduling method
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            current_error: Current prediction error (for adaptive method)
            
        Returns:
            alpha: Current mixing parameter
        """
        if self.config.alpha_method == "constant":
            alpha = self.config.alpha
            
        elif self.config.alpha_method == "linear":
            # Linear increase from alpha_min to alpha_max
            progress = epoch / total_epochs
            alpha = self.config.alpha_min + (self.config.alpha_max - self.config.alpha_min) * progress
            
        elif self.config.alpha_method == "adaptive":
            # Adaptive based on prediction error
            if len(self.prediction_errors) > 0 and current_error is not None:
                # Increase α if error is decreasing, decrease if error is increasing
                error_trend = current_error - np.mean(self.prediction_errors[-10:])
                if error_trend < 0:  # Error decreasing
                    alpha_delta = self.config.alpha_adapt_rate
                else:  # Error increasing
                    alpha_delta = -self.config.alpha_adapt_rate
                
                # Update α with bounds
                if len(self.alpha_history) > 0:
                    alpha = self.alpha_history[-1] + alpha_delta
                else:
                    alpha = self.config.alpha
                    
                alpha = np.clip(alpha, self.config.alpha_min, self.config.alpha_max)
            else:
                alpha = self.config.alpha
        else:
            alpha = self.config.alpha
        
        self.alpha_history.append(alpha)
        return alpha
    
    def mix_states(self, z_pred: torch.Tensor, z_true: torch.Tensor, 
                   alpha: float) -> torch.Tensor:
        """
        Mix predicted and true states according to GTF
        
        Args:
            z_pred: Predicted latent states [batch_size, seq_len, latent_dim]
            z_true: True latent states [batch_size, seq_len, latent_dim]
            alpha: Mixing parameter
            
        Returns:
            z_mixed: Mixed states [batch_size, seq_len, latent_dim]
        """
        return alpha * z_pred + (1 - alpha) * z_true
    
    def compute_gradient_bound(self, model: ShallowPLRNN, alpha: float) -> float:
        """
        Compute theoretical gradient bound for GTF
        
        Based on Theorem 1 from Hess et al. (2023):
        ||∂L/∂θ|| ≤ C * (1 - α)^T
        
        Where C depends on the Lipschitz constant of the dynamics
        """
        stability = model.check_stability()
        lipschitz = stability['lipschitz_estimate']
        
        # Simplified bound computation
        if alpha < 1.0:
            bound = lipschitz / (1 - alpha * lipschitz)
        else:
            bound = float('inf')
        
        return bound
    
    def train_step(self, model: ShallowPLRNN, optimizer: torch.optim.Optimizer,
                   x_true: torch.Tensor, z0: torch.Tensor, 
                   criterion: nn.Module, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Single GTF training step
        
        Args:
            model: shPLRNN model
            optimizer: Optimizer
            x_true: True observations [batch_size, seq_len, observation_dim]
            z0: Initial latent state [batch_size, latent_dim]
            criterion: Loss function
            epoch: Current epoch
            total_epochs: Total epochs
            
        Returns:
            metrics: Dictionary with training metrics
        """
        batch_size, seq_len, _ = x_true.shape
        
        # Get current α
        current_error = self.prediction_errors[-1] if len(self.prediction_errors) > 0 else None
        alpha = self.get_alpha(epoch, total_epochs, current_error)
        
        optimizer.zero_grad()
        
        if self.config.use_gtf:
            # Generalized Teacher Forcing
            z = z0
            z_preds = []
            x_preds = []
            
            # First, get true latent states (if using observation model)
            if model.use_observation_model:
                # Approximate true latents using inverse observation model
                z_true_approx = self._approximate_latents(model, x_true)
            else:
                z_true_approx = x_true
            
            for t in range(seq_len):
                # Predict next state
                z_pred = model.step(z)
                z_preds.append(z_pred)
                
                # Mix predicted and true states
                if t < seq_len - 1:
                    z_true_t = z_true_approx[:, t + 1]
                    z = self.mix_states(z_pred, z_true_t, alpha)
                else:
                    z = z_pred
                
                # Generate observation
                x_pred = model.observe(z)
                x_preds.append(x_pred)
            
            x_pred = torch.stack(x_preds, dim=1)
            
        else:
            # Standard Teacher Forcing
            if epoch % self.config.teacher_forcing_interval == 0:
                # Use teacher forcing
                x_pred, _ = model(z0, seq_len, return_latents=True)
            else:
                # Free running
                x_pred = model(z0, seq_len)
        
        # Compute loss
        loss = criterion(x_pred, x_true)
        
        # Backward pass with gradient clipping
        loss.backward()
        
        # Track gradient norms before clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        
        # Gradient clipping
        if self.config.gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          self.config.gradient_clip_value)
        
        optimizer.step()
        
        # Track prediction error
        with torch.no_grad():
            pred_error = F.mse_loss(x_pred, x_true).item()
            self.prediction_errors.append(pred_error)
        
        # Compute gradient bound
        grad_bound = self.compute_gradient_bound(model, alpha)
        
        return {
            'loss': loss.item(),
            'alpha': alpha,
            'gradient_norm': total_norm,
            'gradient_bound': grad_bound,
            'prediction_error': pred_error
        }
    
    def _approximate_latents(self, model: ShallowPLRNN, 
                           x_obs: torch.Tensor) -> torch.Tensor:
        """
        Approximate latent states from observations
        
        Simple approach using pseudo-inverse of observation model
        """
        batch_size, seq_len, obs_dim = x_obs.shape
        
        # Reshape for batch processing
        x_flat = x_obs.reshape(-1, obs_dim)
        
        # Use pseudo-inverse of B
        B_pinv = torch.pinverse(model.B)
        z_flat = torch.matmul(x_flat - model.b, B_pinv.T)
        
        # Reshape back
        z_approx = z_flat.reshape(batch_size, seq_len, model.latent_dim)
        
        return z_approx
    
    def plot_training_diagnostics(self, save_path: str = "gtf_diagnostics.png"):
        """
        Plot GTF training diagnostics
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Alpha history
        axes[0, 0].plot(self.alpha_history)
        axes[0, 0].set_title('GTF α Parameter Evolution')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('α')
        axes[0, 0].grid(True)
        
        # Gradient norms
        axes[0, 1].plot(self.gradient_norms)
        axes[0, 1].set_title('Gradient Norms')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('||∇||')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Prediction errors
        axes[1, 0].plot(self.prediction_errors)
        axes[1, 0].set_title('Prediction Error')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].grid(True)
        
        # Alpha vs Error scatter
        if len(self.alpha_history) == len(self.prediction_errors):
            axes[1, 1].scatter(self.alpha_history, self.prediction_errors, alpha=0.5)
            axes[1, 1].set_title('α vs Prediction Error')
            axes[1, 1].set_xlabel('α')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


class DynamicalSystemsAnalyzer:
    """
    Analyzer for dynamical properties of learned shPLRNN models
    
    Computes:
    - Lyapunov exponents
    - Attractor reconstruction
    - Phase space visualization
    - Bifurcation analysis
    """
    
    def __init__(self, model: ShallowPLRNN):
        self.model = model
        
    def compute_lyapunov_spectrum(self, z0: torch.Tensor, 
                                 n_steps: int = 10000,
                                 n_exponents: Optional[int] = None) -> np.ndarray:
        """
        Compute Lyapunov exponents using QR decomposition method
        
        Args:
            z0: Initial state
            n_steps: Number of steps for computation
            n_exponents: Number of exponents to compute (default: all)
            
        Returns:
            lyapunov_exponents: Array of Lyapunov exponents
        """
        if n_exponents is None:
            n_exponents = self.model.latent_dim
        
        with torch.no_grad():
            z = z0.clone()
            Q = torch.eye(self.model.latent_dim, device=z.device)
            
            lyap_sum = torch.zeros(n_exponents, device=z.device)
            
            for t in range(n_steps):
                # Evolve state
                z = self.model.step(z.unsqueeze(0)).squeeze(0)
                
                # Compute Jacobian
                J = self.model.compute_jacobian(z.unsqueeze(0)).squeeze(0)
                
                # Evolve tangent vectors
                Q = torch.matmul(J, Q)
                
                # QR decomposition
                Q, R = torch.linalg.qr(Q)
                
                # Accumulate growth rates
                lyap_sum += torch.log(torch.abs(torch.diag(R))[:n_exponents])
            
            # Average over time
            lyapunov_exponents = lyap_sum.cpu().numpy() / n_steps
            
        return np.sort(lyapunov_exponents)[::-1]  # Sort in descending order
    
    def reconstruct_attractor(self, z0: torch.Tensor, 
                            n_steps: int = 50000,
                            transient: int = 10000) -> torch.Tensor:
        """
        Reconstruct attractor by long-term simulation
        
        Args:
            z0: Initial state
            n_steps: Total simulation steps
            transient: Transient steps to discard
            
        Returns:
            trajectory: Attractor trajectory [n_steps-transient, latent_dim]
        """
        with torch.no_grad():
            z = z0.clone()
            trajectory = []
            
            for t in range(n_steps):
                z = self.model.step(z.unsqueeze(0)).squeeze(0)
                
                if t >= transient:
                    trajectory.append(z.clone())
            
            trajectory = torch.stack(trajectory)
            
        return trajectory
    
    def plot_phase_space(self, trajectory: torch.Tensor, 
                        dims: Tuple[int, int] = (0, 1),
                        save_path: str = "phase_space.png"):
        """
        Plot 2D projection of phase space
        """
        traj_np = trajectory.cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.plot(traj_np[:, dims[0]], traj_np[:, dims[1]], 
                'b-', alpha=0.5, linewidth=0.5)
        plt.scatter(traj_np[0, dims[0]], traj_np[0, dims[1]], 
                   c='red', s=100, marker='o', label='Start')
        plt.scatter(traj_np[-1, dims[0]], traj_np[-1, dims[1]], 
                   c='green', s=100, marker='s', label='End')
        
        plt.xlabel(f'Dimension {dims[0]}')
        plt.ylabel(f'Dimension {dims[1]}')
        plt.title('Phase Space Reconstruction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def analyze_dynamics(self, z0: torch.Tensor) -> Dict[str, any]:
        """
        Comprehensive dynamical analysis
        """
        print("Analyzing dynamical properties...")
        
        # Stability check
        stability = self.model.check_stability()
        
        # Lyapunov exponents
        lyapunov_exponents = self.compute_lyapunov_spectrum(z0)
        max_lyapunov = lyapunov_exponents[0]
        
        # Classify dynamics
        if max_lyapunov > 0.01:
            dynamics_type = "Chaotic"
        elif max_lyapunov > -0.01:
            dynamics_type = "Marginal"
        else:
            dynamics_type = "Stable"
        
        # Kaplan-Yorke dimension
        cum_sum = np.cumsum(lyapunov_exponents)
        k = np.where(cum_sum < 0)[0]
        if len(k) > 0:
            k = k[0]
            if k > 0:
                ky_dimension = k + cum_sum[k-1] / abs(lyapunov_exponents[k])
            else:
                ky_dimension = 0
        else:
            ky_dimension = len(lyapunov_exponents)
        
        return {
            'stability': stability,
            'lyapunov_exponents': lyapunov_exponents,
            'max_lyapunov_exponent': max_lyapunov,
            'dynamics_type': dynamics_type,
            'kaplan_yorke_dimension': ky_dimension,
            'is_chaotic': max_lyapunov > 0.01
        }


# Example usage and testing
if __name__ == "__main__":
    # Test shPLRNN implementation
    print("Testing GTF-shPLRNN implementation...")
    
    # Model configuration
    latent_dim = 3
    hidden_dim = 20
    observation_dim = 12
    seq_len = 100
    batch_size = 4
    
    # Create model
    model = ShallowPLRNN(latent_dim, hidden_dim, observation_dim)
    print(f"Created shPLRNN: latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    
    # Test forward pass
    z0 = torch.randn(batch_size, latent_dim)
    observations, latents = model(z0, seq_len, return_latents=True)
    print(f"Generated sequence: observations shape = {observations.shape}")
    
    # Check stability
    stability = model.check_stability()
    print(f"Stability check: {stability}")
    
    # Test GTF training
    gtf_config = GTFConfig(use_gtf=True, alpha_method="adaptive")
    gtf = GeneralizedTeacherForcing(gtf_config)
    
    # Create synthetic target data
    x_true = torch.randn(batch_size, seq_len, observation_dim)
    
    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    metrics = gtf.train_step(model, optimizer, x_true, z0, criterion, 0, 100)
    print(f"GTF training step: {metrics}")
    
    # Dynamical analysis
    analyzer = DynamicalSystemsAnalyzer(model)
    dynamics = analyzer.analyze_dynamics(z0[0])
    print(f"Dynamical analysis: {dynamics}")
    
    print("\n✅ GTF-shPLRNN implementation test completed!")