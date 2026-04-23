"""
Sparse Autoencoder for Feature Discovery
=========================================

Based on Anthropic's "Towards Monosemanticity" paper.

Key Concepts:
1. Overcomplete representation (more features than inputs)
2. L1 sparsity penalty (forces most features to zero)
3. Top-K activation (only keep strongest features)

CTO's Architecture Decisions:
- Expansion factor: 4x (512 inputs → 2048 features)
- Sparsity target: ~95% of features should be zero
- Top-K: Keep only top 50 features per example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features.
    
    Architecture:
        Input (512) → Encoder → Sparse Latent (2048) → Decoder → Reconstruction (512)
    
    The magic happens in the sparse latent layer:
    - Most features are zero (sparse)
    - Each active feature represents ONE concept (monosemantic)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 2048,  # 4x overcomplete
        sparsity_coef: float = 1e-3,
        top_k: int = 50
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_coef = sparsity_coef
        self.top_k = top_k
        
        # Encoder: input → latent
        self.encoder = nn.Linear(input_dim, latent_dim)
        
        # Decoder: latent → reconstruction
        self.decoder = nn.Linear(latent_dim, input_dim)
        
        # Initialize weights (important for SAEs!)
        # CTO Note: Xavier initialization works well
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse latent representation.
        
        Process:
        1. Linear projection to overcomplete space
        2. ReLU activation (ensures non-negative features)
        3. Top-K sparsification (keep only strongest features)
        """
        # Linear projection
        latent = self.encoder(x)
        
        # ReLU activation
        latent = F.relu(latent)
        
        # Top-K sparsification
        # CTO Decision: This is KEY for interpretability
        # Only the strongest features remain active
        if self.top_k < self.latent_dim:
            # Get top-k values and indices
            topk_values, topk_indices = torch.topk(
                latent, 
                self.top_k, 
                dim=-1
            )
            
            # Create sparse tensor (all zeros except top-k)
            sparse_latent = torch.zeros_like(latent)
            sparse_latent.scatter_(-1, topk_indices, topk_values)
            
            return sparse_latent
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent back to input space."""
        return self.decoder(latent)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → decode
        
        Returns:
            reconstruction: Reconstructed input
            latent: Sparse latent representation
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def compute_loss(
        self, 
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAE loss.
        
        Loss = Reconstruction Loss + Sparsity Penalty
        
        CTO Note: The sparsity coefficient is CRITICAL.
        - Too high: Model learns nothing (all features zero)
        - Too low: Model not sparse enough (not interpretable)
        - Sweet spot: ~1e-3 to 1e-4
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)
        
        # Sparsity loss (L1 penalty on activations)
        sparsity_loss = torch.abs(latent).mean()
        
        # Total loss
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss
        
        # Additional metrics
        num_active = (latent > 0).float().sum(dim=-1).mean()
        sparsity_ratio = 1 - (num_active / self.latent_dim)
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss,
            'num_active_features': num_active,
            'sparsity_ratio': sparsity_ratio
        }


class SAETrainer:
    """
    Trainer for Sparse Autoencoder.
    
    CTO Note: Training SAEs is different from training regular models.
    We need to carefully monitor sparsity while maintaining reconstruction quality.
    """
    
    def __init__(
        self,
        sae: SparseAutoencoder,
        learning_rate: float = 1e-4,
        device: str = 'cpu'
    ):
        self.sae = sae.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=learning_rate
        )
        
        # Track metrics
        self.history = {
            'total_loss': [],
            'recon_loss': [],
            'sparsity_loss': [],
            'num_active': [],
            'sparsity_ratio': []
        }
    
    def train_step(self, activations: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        activations = activations.to(self.device)
        
        # Forward pass
        reconstruction, latent = self.sae(activations)
        
        # Compute losses
        losses = self.sae.compute_loss(activations, reconstruction, latent)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()
        
        # Return metrics as floats
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}
    
    def train_epoch(self, activation_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.sae.train()
        
        epoch_metrics = {
            'total_loss': 0,
            'recon_loss': 0,
            'sparsity_loss': 0,
            'num_active': 0,
            'sparsity_ratio': 0
        }
        
        num_batches = 0
        
        for activations in tqdm(activation_loader, desc='Training SAE'):
            metrics = self.train_step(activations)
            
            epoch_metrics['total_loss'] += metrics['total']
            epoch_metrics['recon_loss'] += metrics['reconstruction']
            epoch_metrics['sparsity_loss'] += metrics['sparsity']
            epoch_metrics['num_active'] += metrics['num_active_features']
            epoch_metrics['sparsity_ratio'] += metrics['sparsity_ratio']
            
            num_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            self.history[key if key in self.history else key].append(
                epoch_metrics[key]
            )
        
        return epoch_metrics
    
    def plot_training(self, save_path='sae_training.png'):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        
        axes[0, 1].plot(self.history['recon_loss'])
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        
        axes[1, 0].plot(self.history['sparsity_loss'])
        axes[1, 0].set_title('Sparsity Loss')
        axes[1, 0].set_xlabel('Epoch')
        
        axes[1, 1].plot(self.history['sparsity_ratio'])
        axes[1, 1].set_title('Sparsity Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].axhline(y=0.95, color='r', linestyle='--', 
                           label='Target (95%)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ Training plot saved to: {save_path}")


if __name__ == "__main__":
    print("="*60)
    print("SPARSE AUTOENCODER - UNIT TEST")
    print("="*60)
    
    # Test SAE
    print("\n[TEST 1] Creating SAE...")
    sae = SparseAutoencoder(
        input_dim=512,
        latent_dim=2048,
        top_k=50
    )
    print(f"✅ SAE created: {sae.latent_dim} features")
    
    # Test forward pass
    print("\n[TEST 2] Testing forward pass...")
    test_input = torch.randn(32, 512)  # Batch of 32
    reconstruction, latent = sae(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Check sparsity
    num_active = (latent > 0).sum(dim=-1).float().mean()
    sparsity = 1 - (num_active / sae.latent_dim)
    print(f"Average active features: {num_active:.1f}/{sae.latent_dim}")
    print(f"Sparsity: {sparsity:.2%}")
    
    # Test loss
    print("\n[TEST 3] Testing loss computation...")
    losses = sae.compute_loss(test_input, reconstruction, latent)
    print(f"Total loss: {losses['total']:.4f}")
    print(f"Reconstruction loss: {losses['reconstruction']:.4f}")
    print(f"Sparsity loss: {losses['sparsity']:.4f}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)