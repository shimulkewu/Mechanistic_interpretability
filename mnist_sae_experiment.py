"""
MNIST Sparse Autoencoder Experiment
====================================

This is the MAIN script that ties everything together.

What it does:
1. Load/train MNIST classifier
2. Extract hidden activations from classifier
3. Train SAE on those activations
4. Visualize discovered features

CTO Note: This is your first complete interpretability pipeline!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mnist_classifier import SimpleMNISTClassifier, train_mnist_classifier
from interpreters.sparse_autoencoder import SparseAutoencoder, SAETrainer


def extract_activations(model, data_loader, device='cpu'):
    """
    Extract hidden layer activations from MNIST classifier.
    
    CTO Note: This is the KEY step.
    We're getting the internal representations that the classifier learned.
    These are what we'll interpret with our SAE.
    """
    model.eval()
    all_activations = []
    all_labels = []
    
    print("Extracting activations from classifier...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            
            # Get hidden activations
            _, hidden = model(images, return_hidden=True)
            
            all_activations.append(hidden.cpu())
            all_labels.append(labels)
    
    # Concatenate all batches
    activations = torch.cat(all_activations, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"✅ Extracted {activations.shape[0]} activation vectors")
    print(f"   Shape: {activations.shape}")
    
    return activations, labels


def visualize_features(sae, classifier, test_loader, num_features=10):
    """
    Visualize what each SAE feature learned.
    
    Method:
    1. For each feature, find images that activate it most
    2. Show those images
    3. See what pattern the feature detects!
    
    This is where interpretability becomes VISUAL!
    """
    print(f"\nVisualizing top {num_features} features...")
    
    # Extract activations and get SAE features
    classifier.eval()
    sae.eval()
    
    all_images = []
    all_features = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            _, hidden = classifier(images, return_hidden=True)
            _, features = sae(hidden)
            
            all_images.append(images)
            all_features.append(features)
    
    all_images = torch.cat(all_images, dim=0)
    all_features = torch.cat(all_features, dim=0)
    
    # For each feature, find top activating images
    fig, axes = plt.subplots(num_features, 5, figsize=(15, 3*num_features))
    
    for feature_idx in range(num_features):
        # Get activation strengths for this feature
        activations = all_features[:, feature_idx]
        
        # Find top 5 activating images
        top_indices = torch.topk(activations, k=5).indices
        
        for img_idx, ax in enumerate(axes[feature_idx]):
            image = all_images[top_indices[img_idx]].squeeze()
            activation = activations[top_indices[img_idx]].item()
            
            ax.imshow(image, cmap='gray')
            ax.set_title(f'Act: {activation:.2f}')
            ax.axis('off')
        
        # Label the row
        axes[feature_idx, 0].set_ylabel(
            f'Feature {feature_idx}',
            rotation=0,
            labelpad=50,
            fontsize=12,
            weight='bold'
        )
    
    plt.tight_layout()
    plt.savefig('sae_features_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Feature visualization saved to: sae_features_visualization.png")
    
    return fig


def main():
    """
    Main experiment pipeline.
    
    CTO's Execution Plan:
    1. Train classifier (or load if exists)
    2. Extract activations
    3. Train SAE on activations
    4. Visualize learned features
    5. Analyze what we discovered
    """
    
    print("="*70)
    print(" MNIST SPARSE AUTOENCODER EXPERIMENT")
    print(" Discovering Interpretable Features in Neural Networks")
    print("="*70)
    
    device = 'cpu'  # Use 'cuda' if you have GPU
    
    # ============================================================
    # STEP 1: Get MNIST Classifier
    # ============================================================
    print("\n" + "="*70)
    print("STEP 1: MNIST Classifier")
    print("="*70)
    
    if os.path.exists('mnist_classifier.pth'):
        print("Loading existing classifier...")
        classifier = SimpleMNISTClassifier().to(device)
        classifier.load_state_dict(torch.load('mnist_classifier.pth'))
        
        # Quick accuracy check
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=128)
        
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = classifier(images.to(device))
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels.to(device)).sum().item()
                total += labels.size(0)
        
        print(f"✅ Classifier loaded. Accuracy: {100*correct/total:.2f}%")
    else:
        print("Training new classifier...")
        classifier, train_loader, test_loader = train_mnist_classifier(
            epochs=5,
            device=device
        )
        torch.save(classifier.state_dict(), 'mnist_classifier.pth')
    
    # Load data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # ============================================================
    # STEP 2: Extract Activations
    # ============================================================
    print("\n" + "="*70)
    print("STEP 2: Extracting Hidden Activations")
    print("="*70)
    
    train_activations, train_labels = extract_activations(
        classifier, train_loader, device
    )
    
    test_activations, test_labels = extract_activations(
        classifier, test_loader, device
    )
    
    # Create data loader for activations
    activation_dataset = TensorDataset(train_activations)
    activation_loader = DataLoader(
        activation_dataset,
        batch_size=256,
        shuffle=True
    )
    
    # ============================================================
    # STEP 3: Train Sparse Autoencoder
    # ============================================================
    print("\n" + "="*70)
    print("STEP 3: Training Sparse Autoencoder")
    print("="*70)
    
    # Create SAE
    sae = SparseAutoencoder(
        input_dim=512,
        latent_dim=2048,  # 4x overcomplete
        sparsity_coef=1e-3,
        top_k=50
    )
    
    print(f"SAE Configuration:")
    print(f"  Input dim: {sae.input_dim}")
    print(f"  Latent dim: {sae.latent_dim}")
    print(f"  Expansion: {sae.latent_dim/sae.input_dim:.1f}x")
    print(f"  Top-K: {sae.top_k}")
    print(f"  Sparsity coef: {sae.sparsity_coef}")
    
    # Train SAE
    trainer = SAETrainer(sae, learning_rate=1e-4, device=device)
    
    num_epochs = 20
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(activation_loader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print(f"  Reconstruction: {metrics['recon_loss']:.4f}")
        print(f"  Sparsity: {metrics['sparsity_loss']:.4f}")
        print(f"  Active Features: {metrics['num_active']:.1f}/{sae.latent_dim}")
        print(f"  Sparsity Ratio: {metrics['sparsity_ratio']:.2%}")
    
    # Plot training
    trainer.plot_training('sae_training.png')
    
    # Save SAE
    torch.save(sae.state_dict(), 'mnist_sae.pth')
    print("\n✅ SAE saved to: mnist_sae.pth")
    
    # ============================================================
    # STEP 4: Visualize Features
    # ============================================================
    print("\n" + "="*70)
    print("STEP 4: Visualizing Learned Features")
    print("="*70)
    
    visualize_features(sae, classifier, test_loader, num_features=20)
    
    # ============================================================
    # STEP 5: Analysis
    # ============================================================
    print("\n" + "="*70)
    print("STEP 5: Feature Analysis")
    print("="*70)
    
    # Analyze which features activate for which digits
    sae.eval()
    classifier.eval()
    
    feature_digit_activation = torch.zeros(sae.latent_dim, 10)
    
    with torch.no_grad():
        for images, labels in test_loader:
            _, hidden = classifier(images.to(device), return_hidden=True)
            _, features = sae(hidden)
            
            for digit in range(10):
                mask = labels == digit
                if mask.sum() > 0:
                    digit_features = features[mask]
                    feature_digit_activation[:, digit] += digit_features.sum(dim=0).cpu()
    
    # Find most digit-specific features
    print("\nMost digit-specific features:")
    for digit in range(10):
        feature_idx = feature_digit_activation[:, digit].argmax()
        activation = feature_digit_activation[feature_idx, digit].item()
        print(f"  Digit {digit}: Feature {feature_idx} (activation: {activation:.0f})")
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\n📊 Results saved:")
    print("  - mnist_classifier.pth (trained classifier)")
    print("  - mnist_sae.pth (trained SAE)")
    print("  - sae_training.png (training curves)")
    print("  - sae_features_visualization.png (feature visualizations)")
    print("\n🎯 Next steps:")
    print("  1. Open sae_features_visualization.png")
    print("  2. Look at what each feature learned!")
    print("  3. Notice how features detect specific patterns (curves, lines, etc.)")
    print("\n💡 This is interpretability in action!")


if __name__ == "__main__":
    main()