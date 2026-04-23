"""
Simple MNIST Classifier
======================

We need a trained model to get activations from.
This is just a basic neural network - nothing fancy.

CTO Note: We're not trying to get SOTA accuracy.
We just need a model that works well enough to have interesting activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleMNISTClassifier(nn.Module):
    """
    Simple 2-layer neural network for MNIST
    
    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden: 512 neurons with ReLU
    - Output: 10 classes
    
    CTO Decision: 512 hidden units is a sweet spot
    - Large enough to learn good features
    - Small enough to train fast
    - Will give us interesting activations to interpret
    """
    
    def __init__(self, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.hidden_size = hidden_size
        
    def forward(self, x, return_hidden=False):
        """
        Forward pass with option to return hidden activations
        
        Args:
            x: Input images [batch, 1, 28, 28]
            return_hidden: If True, return (logits, hidden_activations)
        """
        # Flatten images
        x = x.view(-1, 28 * 28)
        
        # Hidden layer with ReLU
        hidden = F.relu(self.fc1(x))
        
        # Output layer
        logits = self.fc2(hidden)
        
        if return_hidden:
            return logits, hidden
        return logits


def train_mnist_classifier(epochs=5, batch_size=128, device='cpu'):
    """
    Train the MNIST classifier
    
    CTO Note: 5 epochs is enough. We don't need perfect accuracy.
    We just need a model that learns SOMETHING.
    """
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    model = SimpleMNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch+1}: Test Accuracy: {test_acc:.2f}%')
    
    print("\n✅ Training complete!")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    return model, train_loader, test_loader


if __name__ == "__main__":
    print("="*60)
    print("TRAINING MNIST CLASSIFIER")
    print("="*60)
    
    # Train model
    model, train_loader, test_loader = train_mnist_classifier(
        epochs=5,
        device='cpu'  # Use 'cuda' if you have GPU
    )
    
    # Save model
    torch.save(model.state_dict(), 'mnist_classifier.pth')
    print("\n✅ Model saved to: mnist_classifier.pth")
    
    # Visualize some predictions
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Show first 10 predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.set_title(f'Pred: {predicted[idx]}\nTrue: {labels[idx]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    print("✅ Predictions saved to: mnist_predictions.png")