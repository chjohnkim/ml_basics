import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class Normalizer:
    def __init__(self, X, device):
        self.mean = X.mean(dim=0).to(device)
        self.std = X.std(dim=0).to(device)

    def normalize(self, X):
        return (X - self.mean) / self.std

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=None):
        super().__init__()

        layers = []
        
        # Input layer → First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer (no activation here - handled by loss function)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)

def train_model(model, normalizer, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        # TRAINING PHASE
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 0. Normalize input data
            if normalizer is not None:
                batch_x = normalizer.normalize(batch_x)
            # 1. Forward Pass
            outputs = model(batch_x)
            # 2. Compute loss
            loss = criterion(outputs, batch_y)
            # 3. Clear previous gradeints
            optimizer.zero_grad()
            # 4. Backward pass
            loss.backward()
            # 5. Update weights
            optimizer.step()

            # Track metrics
            batch_size = batch_x.size(0)
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, dim=1)
            train_total += batch_size
            train_correct += (predicted == batch_y).sum().item()

        # Average training metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
    
        # VALIDATION PHASE
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                # 0. Normalize input data
                if normalizer is not None:
                    batch_x = normalizer.normalize(batch_x)
                # 1. Forward Pass
                outputs = model(batch_x)
                # 2. Compute loss
                loss = criterion(outputs, batch_y)
            
                # Track metrics
                batch_size = batch_x.size(0)
                val_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, dim=1)
                val_total += batch_size
                val_correct += (predicted == batch_y).sum().item()
        # Average training metrics
        val_loss /= val_total
        val_acc = val_correct / val_total

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        if (epoch+1)%10==0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return history

def evaluate_model(model, normalizer, test_loader, device): 
    model.eval()
    test_total = 0
    test_correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 0. Normalize input data
            if normalizer is not None:
                batch_x = normalizer.normalize(batch_x)
            # Forward pass
            outputs = model(batch_x)
            # Track metrics
            batch_size = batch_x.size(0)
            _, predicted = torch.max(outputs, dim=1)
            test_total += batch_size
            test_correct += (predicted == batch_y).sum().item()
    # Average training metrics
    test_acc = test_correct / test_total
    print(f'\nTest Accuracy: {test_acc:.4f}')

def plot_training_history(history):
    """
    Visualize training history.
    
    Args:
        history (dict): Training history from train_model()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # ========================================================================
    # 0. Define hyperparameters
    # ========================================================================
    # Train params
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-3
    # Model params
    input_dim = 20 # 20 features
    output_dim = 3 # num classes
    hidden_dims = [32, 32, 32]
    # Regularization params
    weight_decay = 1e-5
    dropout_rate = 0.3
    normalize = True


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # ========================================================================
    # 1. CREATE SYNTHETIC DATASET
    # ========================================================================
    print('\n1. Creating synthetic dataset...')
    X, y = make_classification(
        n_samples=1000,      # Total samples
        n_features=input_dim,# Number of features
        n_informative=15,    # Informative features
        n_redundant=5,       # Redundant features
        n_classes=output_dim,# Number of classes
        random_state=42
    )

    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f'Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}')
    
    # ========================================================================
    # 2. CONVERT TO PYTORCH TENSORS AND CREATE DATALOADERS
    # ========================================================================
    print('\n2. Creating PyTorch DataLoaders...')
    
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.LongTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute normalization params
    if normalize:
        normalizer = Normalizer(X_train, device)
        print(f'Mean: {normalizer.mean.shape}, Std: {normalizer.std.shape}')
    else:
        normalizer = None

    # ========================================================================
    # 3. CREATE MODEL
    # ========================================================================
    print('\n3. Creating MLP model...')
    model = MLP(input_dim, hidden_dims, output_dim, dropout_rate)
    model.to(device)
    print(model)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

    # ========================================================================
    # 4. DEFINE LOSS AND OPTIMIZER
    # ========================================================================
    print('\n4. Setting up training...')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ========================================================================
    # 5. TRAIN MODEL
    # ========================================================================
    print('\n5. Training model...')
    history = train_model(model, normalizer, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # ========================================================================
    # 6. EVALUATE ON TEST SET
    # ========================================================================
    print('\n6. Evaluating on test set...')
    evaluate_model(model, normalizer, test_loader, device)
    
    # ========================================================================
    # 7. VISUALIZE RESULTS
    # ========================================================================
    print('\n7. Plotting training history...')
    plot_training_history(history)
    
if __name__=="__main__":
    main()