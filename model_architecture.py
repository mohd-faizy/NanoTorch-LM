import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from hyperparameters import ModelConfig

# =============================================================================
# 1. Self-Attention Head
# =============================================================================
class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention.
    
    Self-attention is the mechanism that allows the model to look at different positions 
    of the input sequence to compute a representation of the sequence.
    """
    def __init__(self, embedding_dim: int, block_size: int, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape 
        
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        # Apply mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Apply softmax
        wei = F.softmax(wei, dim=-1)
        
        # Aggregate values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out

# =============================================================================
# 2. Multi-Head Attention
# =============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.n_heads
        self.head_size = config.embedding_dim // config.n_heads
        
        self.heads = nn.ModuleList([
            SelfAttentionHead(config.embedding_dim, config.block_size, self.head_size) 
            for _ in range(self.num_heads)
        ])
        
        self.proj = nn.Linear(self.num_heads * self.head_size, config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# =============================================================================
# 3. Feed Forward Network
# =============================================================================
class FeedForward(nn.Module):
    """
    A simple multilayer perceptron (MLP).
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# 4. Transformer Block
# =============================================================================
class Block(nn.Module):
    """
    A single Transformer block.
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# =============================================================================
# 5. The Tiny GPT Model
# =============================================================================
class TinyGPT(nn.Module):
    """
    The main GPT model architecture.
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size) 

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape 
        
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
            
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new text given a starting context `idx`.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
