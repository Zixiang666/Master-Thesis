#!/usr/bin/env python3
"""
Corrected GTF-shPLRNN Implementation
===================================

This module provides a corrected implementation of the GTF-enhanced shallow PLRNN
based on the TRR_WS reference implementation, specifically designed for ECG
classification tasks.

Key improvements:
- Proper shallow PLRNN architecture with correct dimensionality
- GTF regularization with accurate Jacobian computation
- Adaptive alpha scheduling
- ECG-specific input/output handling
- Integration with experiment management system

Reference: TRR_WS project shallow PLRNN implementation
Author: Master Thesis Project
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class ShallowPLRNN(nn.Module):
    """
    Shallow Piecewise Linear Recurrent Neural Network
    
    Based on the TRR_WS implementation, adapted for ECG classification.
    
    Architecture:
    z_{t+1} = A * z_t + W1 * ReLU(W2 * z_t + h2) + h1 + C * s_t
    
    where:
    - z_t: latent state at time t
    - s_t: external input at time t
    - A: diagonal transition matrix
    - W1, W2: weight matrices for nonlinear transformation
    - h1, h2: bias vectors
    - C: input weight matrix
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int = 32,
        clip_range: Optional[float] = None,
        layer_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.clip_range = clip_range
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        
        # Initialize parameters following TRR_WS structure
        self._initialize_parameters()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        logger.info(f"ShallowPLRNN initialized: input_dim={input_dim}, latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    
    def _initialize_parameters(self):
        """Initialize model parameters following TRR_WS conventions"""
        
        # Diagonal transition matrix A
        self.A = nn.Parameter(torch.zeros(self.latent_dim))
        
        # Nonlinear transformation matrices
        self.W1 = nn.Parameter(torch.zeros(self.latent_dim, self.hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(self.hidden_dim, self.latent_dim))
        
        # Bias vectors
        self.h1 = nn.Parameter(torch.zeros(self.latent_dim))
        self.h2 = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Input connection matrix
        self.C = nn.Parameter(torch.zeros(self.latent_dim, self.input_dim))
        
        # Output projection
        self.output_projection = nn.Linear(self.latent_dim, self.output_dim)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Reset parameters with appropriate initialization"""
        
        # Initialize A with small random values around identity
        nn.init.uniform_(self.A, -0.1, 0.1)
        
        # Initialize transformation matrices
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        
        # Initialize biases
        nn.init.zeros_(self.h1)
        nn.init.zeros_(self.h2)
        
        # Initialize input connections
        nn.init.xavier_uniform_(self.C)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward_step(self, z: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single forward step of the shallow PLRNN
        
        Args:
            z: Current latent state [batch_size, latent_dim]
            s: External input [batch_size, input_dim] (optional)
            
        Returns:
            Next latent state [batch_size, latent_dim]
        """
        
        # Linear transition: A * z
        next_z = z * self.A  # Element-wise multiplication (A is diagonal)
        
        # Nonlinear transformation: W1 * ReLU(W2 * z + h2)
        hidden = torch.einsum('bij,bj->bi', self.W2.unsqueeze(0).expand(z.size(0), -1, -1), z) + self.h2
        hidden_activated = F.relu(hidden)
        nonlinear_term = torch.einsum('bij,bj->bi', self.W1.unsqueeze(0).expand(z.size(0), -1, -1), hidden_activated)
        
        next_z = next_z + nonlinear_term + self.h1
        
        # External input: C * s
        if s is not None and self.C is not None:
            input_term = torch.einsum('bij,bj->bi', self.C.unsqueeze(0).expand(z.size(0), -1, -1), s)
            next_z = next_z + input_term
        
        # Apply clipping if specified
        if self.clip_range is not None:
            next_z = torch.clamp(next_z, -self.clip_range, self.clip_range)
        
        # Apply layer normalization if specified
        if self.layer_norm:
            next_z = next_z - next_z.mean(dim=1, keepdim=True)
        
        return next_z
    
    def compute_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix for GTF regularization
        
        Args:
            z: Current state [batch_size, latent_dim]
            
        Returns:
            Jacobian [batch_size, latent_dim, latent_dim]
        """
        
        batch_size = z.size(0)
        
        # Compute which hidden units are active: ReLU derivative
        hidden_input = torch.einsum('bij,bj->bi', self.W2.unsqueeze(0).expand(batch_size, -1, -1), z) + self.h2
        active_mask = (hidden_input > 0).float()  # ReLU derivative
        
        # Jacobian computation: dz_{t+1}/dz_t
        # J = A + W1 * diag(active_mask) * W2
        jacobian = torch.diag_embed(self.A).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add nonlinear term contribution
        W1_expanded = self.W1.unsqueeze(0).expand(batch_size, -1, -1)
        W2_expanded = self.W2.unsqueeze(0).expand(batch_size, -1, -1)
        
        # W1 * diag(active_mask) * W2
        nonlinear_jacobian = torch.einsum('bih,bh,bhj->bij', W1_expanded, active_mask, W2_expanded)
        jacobian = jacobian + nonlinear_jacobian
        
        return jacobian
    
    def forward(self, input_sequence: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the entire sequence
        
        Args:
            input_sequence: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim] for single step
            initial_state: Initial latent state [batch_size, latent_dim] (optional)
            
        Returns:
            Output logits [batch_size, output_dim]
        """
        
        batch_size = input_sequence.size(0)
        
        # Handle different input dimensions
        if input_sequence.dim() == 2:
            # Single time step input
            input_sequence = input_sequence.unsqueeze(1)
        
        seq_len = input_sequence.size(1)
        
        # Initialize latent state
        if initial_state is None:
            z = torch.zeros(batch_size, self.latent_dim, device=input_sequence.device)
        else:
            z = initial_state
        
        # Process sequence
        for t in range(seq_len):
            s_t = input_sequence[:, t, :]  # Current input
            z = self.forward_step(z, s_t)
            
            # Apply dropout during training
            if self.training and self.dropout is not None:
                z = self.dropout(z)
        
        # Generate output from final state
        output = self.output_projection(z)
        
        return output


class GTFRegularizer:
    """
    Generalized Teacher Forcing (GTF) regularizer
    
    Implements GTF regularization by constraining the spectral norm of Jacobians
    """
    
    def __init__(
        self,
        alpha: float = 0.2,
        adaptive: bool = False,
        update_frequency: int = 10,
        estimation_method: str = 'geometric_mean'
    ):
        self.alpha = alpha
        self.adaptive = adaptive
        self.update_frequency = update_frequency
        self.estimation_method = estimation_method
        self.step_count = 0
        
        logger.info(f"GTF Regularizer initialized: alpha={alpha}, adaptive={adaptive}")
    
    def compute_gtf_loss(
        self,
        model: ShallowPLRNN,
        latent_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GTF regularization loss
        
        Args:
            model: ShallowPLRNN model
            latent_states: Sequence of latent states [batch_size, seq_len, latent_dim]
            
        Returns:
            GTF loss scalar
        """
        
        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)
        
        batch_size, seq_len, latent_dim = latent_states.shape
        
        gtf_loss = 0.0
        
        # Compute loss over sequence (excluding first time step)
        for t in range(1, seq_len):
            z_t = latent_states[:, t, :]
            
            # Compute Jacobian at current state
            jacobian = model.compute_jacobian(z_t)
            
            # Compute spectral norm (largest singular value)
            spectral_norms = torch.linalg.matrix_norm(jacobian, ord=2, dim=(1, 2))
            
            # GTF constraint: spectral norm should be <= 1/alpha - 1
            target_spectral_norm = 1.0 / self.alpha - 1.0
            
            # Loss: max(0, spectral_norm - target)^2
            excess = torch.clamp(spectral_norms - target_spectral_norm, min=0.0)
            gtf_loss += torch.mean(excess ** 2)
        
        return gtf_loss / (seq_len - 1) if seq_len > 1 else gtf_loss
    
    def update_alpha(
        self,
        model: ShallowPLRNN,
        latent_states: torch.Tensor
    ) -> float:
        """
        Update alpha adaptively based on current spectral norms
        
        Args:
            model: ShallowPLRNN model
            latent_states: Sequence of latent states
            
        Returns:
            Updated alpha value
        """
        
        if not self.adaptive:
            return self.alpha
        
        self.step_count += 1
        
        if self.step_count % self.update_frequency != 0:
            return self.alpha
        
        # Compute current spectral norms
        with torch.no_grad():
            if latent_states.dim() == 2:
                latent_states = latent_states.unsqueeze(1)
            
            spectral_norms = []
            
            for t in range(latent_states.size(1)):
                z_t = latent_states[:, t, :]
                jacobian = model.compute_jacobian(z_t)
                norms = torch.linalg.matrix_norm(jacobian, ord=2, dim=(1, 2))
                spectral_norms.append(norms)
            
            spectral_norms = torch.stack(spectral_norms, dim=1)  # [batch, time]
            
            # Estimate appropriate alpha based on current spectral norms
            if self.estimation_method == 'geometric_mean':
                mean_norm = torch.exp(torch.mean(torch.log(spectral_norms + 1e-8)))
            elif self.estimation_method == 'arithmetic_mean':
                mean_norm = torch.mean(spectral_norms)
            else:  # max
                mean_norm = torch.max(spectral_norms)
            
            # Update alpha: alpha = 1 - 1/spectral_norm
            estimated_alpha = max(0.01, 1.0 - 1.0 / (mean_norm.item() + 1e-8))
            
            # Exponential moving average
            self.alpha = 0.9 * self.alpha + 0.1 * estimated_alpha
            
            logger.debug(f"Updated alpha: {self.alpha:.4f} (estimated: {estimated_alpha:.4f})")
        
        return self.alpha


class GTFShallowPLRNN(nn.Module):
    """
    Complete GTF-enhanced Shallow PLRNN for ECG Classification
    
    Combines the shallow PLRNN with GTF regularization and ECG-specific preprocessing
    """
    
    def __init__(
        self,
        input_channels: int = 12,
        sequence_length: int = 5000,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        alpha: float = 0.2,
        adaptive_alpha: bool = False,
        gtf_weight: float = 1.0,
        preprocessing: str = 'conv',
        clip_range: Optional[float] = None,
        layer_norm: bool = False,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gtf_weight = gtf_weight
        self.preprocessing = preprocessing
        
        # Input preprocessing
        if preprocessing == 'conv':
            self.input_processor = self._create_conv_preprocessor()
            preprocessed_dim = 128  # Output from conv preprocessor
        elif preprocessing == 'linear':
            self.input_processor = nn.Linear(input_channels * sequence_length, latent_dim)
            preprocessed_dim = latent_dim
        else:  # 'none'
            self.input_processor = None
            preprocessed_dim = input_channels * sequence_length
        
        # Core PLRNN model
        self.plrnn = ShallowPLRNN(
            input_dim=preprocessed_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            clip_range=clip_range,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate
        )
        
        # GTF regularizer
        self.gtf_regularizer = GTFRegularizer(
            alpha=alpha,
            adaptive=adaptive_alpha
        )
        
        logger.info(f"GTF-shPLRNN initialized: preprocessing={preprocessing}, alpha={alpha}")
    
    def _create_conv_preprocessor(self) -> nn.Module:
        """Create convolutional preprocessor for ECG signals"""
        
        return nn.Sequential(
            # First conv layer: extract local features
            nn.Conv1d(self.input_channels, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Second conv layer: intermediate features
            nn.Conv1d(32, 64, kernel_size=9, stride=3, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Third conv layer: high-level features
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess ECG input signals
        
        Args:
            x: Input tensor [batch_size, channels, sequence_length]
            
        Returns:
            Preprocessed tensor
        """
        
        if self.input_processor is None:
            # Flatten input
            return x.view(x.size(0), -1)
        elif self.preprocessing == 'conv':
            # Apply convolutional preprocessing
            return self.input_processor(x)
        else:  # linear
            # Flatten and apply linear transformation
            x_flat = x.view(x.size(0), -1)
            return self.input_processor(x_flat)
    
    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with GTF regularization
        
        Args:
            x: Input ECG signals [batch_size, channels, sequence_length]
            return_latent: Whether to return latent states
            
        Returns:
            Dictionary containing:
            - 'logits': Output logits [batch_size, output_dim]
            - 'gtf_loss': GTF regularization loss
            - 'latent_states': Latent states (if return_latent=True)
        """
        
        batch_size = x.size(0)
        
        # Preprocess input
        processed_input = self.preprocess_input(x)
        
        # Forward through PLRNN
        logits = self.plrnn(processed_input)
        
        # Collect latent states for GTF computation
        if self.training or return_latent:
            # Re-run forward pass to collect intermediate states
            latent_states = self._collect_latent_states(processed_input)
            
            # Compute GTF loss
            gtf_loss = self.gtf_regularizer.compute_gtf_loss(self.plrnn, latent_states)
            
            # Update alpha if adaptive
            if self.training:
                self.gtf_regularizer.update_alpha(self.plrnn, latent_states)
        else:
            gtf_loss = torch.tensor(0.0, device=x.device)
            latent_states = None
        
        result = {
            'logits': logits,
            'gtf_loss': gtf_loss
        }
        
        if return_latent:
            result['latent_states'] = latent_states
        
        return result
    
    def _collect_latent_states(self, processed_input: torch.Tensor) -> torch.Tensor:
        """Collect latent states during forward pass for GTF computation"""
        
        batch_size = processed_input.size(0)
        
        # Initialize latent state
        z = torch.zeros(batch_size, self.latent_dim, device=processed_input.device)
        latent_states = [z.clone()]
        
        # Simple approach: treat processed input as single time step
        # For more complex sequences, this could be modified
        z = self.plrnn.forward_step(z, processed_input)
        latent_states.append(z)
        
        return torch.stack(latent_states, dim=1)  # [batch, time, latent_dim]
    
    def compute_total_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        gtf_loss: torch.Tensor,
        criterion: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including GTF regularization
        
        Args:
            predictions: Model predictions [batch_size, output_dim]
            targets: Target labels [batch_size, output_dim]
            gtf_loss: GTF regularization loss
            criterion: Loss criterion (e.g., BCEWithLogitsLoss)
            
        Returns:
            Dictionary containing loss components
        """
        
        # Main task loss
        task_loss = criterion(predictions, targets)
        
        # Total loss
        total_loss = task_loss + self.gtf_weight * gtf_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'gtf_loss': gtf_loss,
            'gtf_weight': self.gtf_weight,
            'current_alpha': self.gtf_regularizer.alpha
        }
    
    @property
    def current_alpha(self) -> float:
        """Get current alpha value"""
        return self.gtf_regularizer.alpha
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information for logging"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'GTF-shPLRNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'current_alpha': self.current_alpha,
            'gtf_weight': self.gtf_weight,
            'preprocessing': self.preprocessing
        }


# Factory function for easy model creation
def create_gtf_shplrnn(
    config: Dict[str, any],
    input_channels: int = 12,
    sequence_length: int = 5000,
    output_dim: int = 32
) -> GTFShallowPLRNN:
    """
    Factory function to create GTF-shPLRNN with configuration
    
    Args:
        config: Model configuration dictionary
        input_channels: Number of ECG channels
        sequence_length: Length of ECG sequences
        output_dim: Number of output classes
        
    Returns:
        Configured GTF-shPLRNN model
    """
    
    return GTFShallowPLRNN(
        input_channels=input_channels,
        sequence_length=sequence_length,
        latent_dim=config.get('latent_dim', 32),
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=output_dim,
        alpha=config.get('alpha', 0.2),
        adaptive_alpha=config.get('adaptive_alpha', False),
        gtf_weight=config.get('gtf_weight', 1.0),
        preprocessing=config.get('preprocessing', 'conv'),
        clip_range=config.get('clip_range', None),
        layer_norm=config.get('layer_norm', False),
        dropout_rate=config.get('dropout_rate', 0.1)
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model creation
    config = {
        'latent_dim': 32,
        'hidden_dim': 64,
        'alpha': 0.2,
        'adaptive_alpha': True,
        'gtf_weight': 1.0,
        'preprocessing': 'conv'
    }
    
    model = create_gtf_shplrnn(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 12, 5000)  # ECG input
    
    with torch.no_grad():
        output = model(x, return_latent=True)
        
        print(f"Input shape: {x.shape}")
        print(f"Output logits shape: {output['logits'].shape}")
        print(f"GTF loss: {output['gtf_loss'].item():.6f}")
        print(f"Model info: {model.get_model_info()}")
        
        if 'latent_states' in output:
            print(f"Latent states shape: {output['latent_states'].shape}")
    
    print("GTF-shPLRNN model test completed successfully!")
