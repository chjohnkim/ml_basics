import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class MinimalCNN(nn.Module):
    """
    Minimal CNN for image classification.
    
    Architecture:
        Input (28×28×1)
        → Conv(32, 3×3) → ReLU → MaxPool(2×2)
        → Conv(64, 3×3) → ReLU → MaxPool(2×2)
        → Flatten → FC(128) → ReLU → Dropout(0.5) → FC(10)
    """

    def __init__(self, num_classes, in_channels):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 2 pools: 28 → 14 → 7, so 7×7×64 = 3136
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_x.size(0)
            correct += (predicted==batch_y).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        running_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                running_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total += batch_x.size(0)
                correct += (predicted==batch_y).sum().item()
            test_loss = running_loss / total
            test_acc = correct / total
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    return history

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, device, num_images=10):
    """Visualize predictions."""
    model.eval()
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i].cpu().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = MinimalCNN(num_classes=10, in_channels=1).to(device)
    print(model)
    print(f'\nTtotal parameters: {sum(p.numel() for p in model.parameters()):,}\n')

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Train
    print('Training...')
    history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

    # Visualize
    print('\nPlotting results...')
    plot_history(history)
    visualize_predictions(model, test_loader, device)

if __name__=="__main__":
    main()