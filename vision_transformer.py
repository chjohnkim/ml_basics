import torch
import torch.nn as nn
import torchvision
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, n_channels):
        super().__init__()
        self.d_model = d_model 
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.linear = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear(x) # B, D, PW, PH
        x = x.flatten(2) # B, D, PW*PH
        x = x.transpose(-2 , -1) # B, P, D
        return x

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()
    self.d_model = d_model
    self.max_seq_length = max_seq_length
    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    pe = torch.zeros(1, max_seq_length, d_model)
    for pos in range(max_seq_length):
        for i in range(d_model):
            if i%2==0:
                pe[:, pos, i] = np.sin(pos/(10000**(2*i/d_model)))
            else:
                pe[:, pos, i] = np.cos(pos/(10000**(2*(i-1)/d_model)))
    self.register_buffer('pe', pe)  # Register as buffer, not parameter


  def forward(self, x):
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = x + self.pe
    return x

class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = torch.matmul(q, k.transpose(-2, -1)) / self.head_size**0.5
        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, v)
        return attention
  
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model),
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.embedding = PatchEmbedding(d_model, patch_size, n_channels)
        img_w, img_h = img_size
        patch_w, patch_h = patch_size
        max_seq_length = img_w//patch_w * img_h//patch_h + 1
        self.pe = PositionalEncoding(d_model, max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads, r_mlp=4) for _ in range(n_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, n_classes),
            #nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0]) # Only take the classification token
        return x


def train(model, train_loader, test_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for img, label in tqdm(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()            
            x = model(img)
            loss = criterion(x, label)
            loss.backward()
            optimizer.step()
            training_loss+=loss.item()
        print(f"Epoch {epoch}: Training Loss: {training_loss/len(train_loader)}")

        model.eval()
        testing_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                x = model(img)
                loss = criterion(x, label)
                prediction = torch.argmax(x, dim=-1)
                correct+=torch.sum(prediction==label).item()
                total+=len(prediction)
                testing_loss+=loss.item()
        print(f"Epoch {epoch}: Testing Loss: {testing_loss/len(test_loader)}, Testing Accuracy: {correct/total}")
    return model 

if __name__=="__main__":
    d_model = 9
    n_classes = 10
    img_size = (32,32)
    patch_size = (16,16)
    n_channels = 1
    n_heads = 3
    n_layers = 3
    batch_size = 128
    epochs = 20
    lr = 0.005
    
    # Load Dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.ToTensor()
    ])

    train_set = MNIST(
        root="./datasets", train=True, download=True, transform=transform
    )
    test_set = MNIST(
        root="./datasets", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViT(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train(model, train_loader, test_loader, optimizer, criterion, epochs, device)