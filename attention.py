import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query matrix (batch_size, n_heads, seq_len_q, d_k)
            K: Key matrix (batch_size, n_heads, seq_len_k, d_k)
            V: Value matrix (batch_size, n_heads, seq_len_v, d_v)
            mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
        
        Returns:
            output: (batch_size, n_heads, seq_len_q, d_v)
            attention_weights: (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        d_k = Q.size(-1)
        
        # Step 1 & 2: Compute scores and scale
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        
        # Step 3: Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Weighted sum of values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, seq_len_q, d_model)
            K: (batch_size, seq_len_k, d_model)
            V: (batch_size, seq_len_v, d_model)
            mask: (batch_size, 1, seq_len_q, seq_len_k) or None
        """
        batch_size = Q.size(0)
        
        # Linear projections and split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention_weights = self.attention(Q, K, V, mask)
        # x: (batch_size, n_heads, seq_len_q, d_k)
        
        # Concatenate heads
        # (batch_size, n_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, n_heads, d_k)
        # -> (batch_size, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(x)
        
        return output, attention_weights


# Test Example
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    print("=" * 60)
    print("MULTI-HEAD ATTENTION TEST")
    print("=" * 60)
    
    # Create random input
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shapes:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Forward pass (self-attention: Q=K=V)
    output, attention_weights = mha(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Verify attention weights sum to 1
    print(f"\nAttention weights sum (should be ~1.0): {attention_weights[0, 0, 0].sum().item():.6f}")
    
    # Test with causal mask (for autoregressive generation)
    print("\n" + "=" * 60)
    print("TESTING WITH CAUSAL MASK")
    print("=" * 60)
    
    # Create causal mask: upper triangular matrix of zeros
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    # Shape: (1, 1, seq_len, seq_len)
    
    output_masked, attention_masked = mha(Q, K, V, mask=causal_mask)
    
    print(f"\nMasked output shape: {output_masked.shape}")
    print(f"\nFirst head attention pattern (position 0 can only attend to itself):")
    print(attention_masked[0, 0, 0, :5])  # First 5 positions
    print(f"\nFirst head attention pattern (position 5 can attend to positions 0-5):")
    print(attention_masked[0, 0, 5, :10])
    
    # Visualize attention pattern
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION")
    print("=" * 60)
    
    # Simple visualization of attention weights for first sample, first head
    attn_matrix = attention_masked[0, 0].detach().numpy()
    print("\nCausal attention pattern (first head, first sample):")
    print("Rows=query positions, Cols=key positions")
    print("(Each row sums to 1.0, values above diagonal are 0 due to mask)\n")
    
    # Print first 6x6 block
    for i in range(min(6, seq_len)):
        row = " ".join([f"{attn_matrix[i, j]:.2f}" for j in range(min(6, seq_len))])
        print(f"Q{i}: {row}")
    
    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n" + "=" * 60)
    print(f"Total parameters in MultiHeadAttention: {total_params:,}")
    print("=" * 60)