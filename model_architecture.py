import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Self-Attention Head
# =============================================================================
class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention.
    
    Self-attention is the mechanism that allows the model to look at different positions 
    of the input sequence to compute a representation of the sequence.
    
    In a "single head", the model learns one set of relationships.
    """
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        # Key, Query, and Value projections:
        # These are linear layers that transform the input vector into three different vectors:
        # - Query (Q): What am I looking for?
        # - Key (K): What do I contain?
        # - Value (V): What information do I pass along if I am matched?
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
        # 'tril' is a lower triangular matrix used for masking.
        # It ensures that when predicting the token at position t, the model can 
        # only attend to tokens at positions 0 to t (past and present), not t+1 (future).
        # We register it as a buffer because it's part of the state but not a learnable parameter.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x input size: (Batch_Size, Time_Steps, Channels/Embedding_Size)
        B, T, C = x.shape 
        
        # 1. Calculate Key and Query vectors
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 2. Compute attention scores (affinities)
        # Calculates how much each token should specific focus on every other token.
        # We divide by sqrt(head_size) (C**0.5) to scale the values. 
        # This prevents the dot products from getting too large, which would make the softmax gradients small.
        wei = q @ k.transpose(-2, -1) / (C ** 0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        # 3. Apply the mask
        # We replace the values where tril is 0 (upper triangle) with -infinity.
        # When we take the softmax, these -inf values will become 0, effectively ignoring future tokens.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 4. Apply softmax
        # Normalizes the scores so they sum to 1 for each position.
        # This gives us a probability distribution over the allowed tokens.
        wei = F.softmax(wei, dim=-1)
        
        # 5. Aggregate the values
        # We calculate the weighted sum of the Value vectors.
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out

# =============================================================================
# 2. Multi-Head Attention
# =============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.
    
    This allows the model to attend to information from different representation 
    subspaces at different positions. For example, one head might focus on 
    syntax, while another focuses on semantics.
    """
    def __init__(self, embedding_dim, block_size, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        
        # Create a list of independent attention heads.
        # They will run in parallel.
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, block_size, head_size) for _ in range(num_heads)])
        
        # Final projection layer.
        # This combines the outputs of all heads back into the original embedding dimension.
        # It allows the heads to communicate their results.
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)

    def forward(self, x):
        # Concatenate the outputs from all heads along the last dimension (channels).
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back to the original embedding dimension.
        return self.proj(out)

# =============================================================================
# 3. Feed Forward Network
# =============================================================================
class FeedForward(nn.Module):
    """
    A simple multilayer perceptron (MLP) applied to each position separately and identically.
    
    It allows the model to "think" about the information gathered by the attention heads.
    It adds non-linearity and computation depth to the model.
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expand the dimension (usually by factor of 4)
            nn.ReLU(),                     # Activation function (Rectified Linear Unit)
            nn.Linear(4 * n_embd, n_embd), # Project back to original dimension
            nn.Dropout(dropout)            # Regularization to prevent overfitting
        )
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 4. Transformer Block
# =============================================================================
class Block(nn.Module):
    """
    A single Transformer block.
    
    Structure:
    Input -> LayerNorm -> Multi-Head Attention -> + (Residual) -> LayerNorm -> FeedForward -> + (Residual)
    """
    def __init__(self, embedding_dim, block_size, n_heads, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim, dropout)
        
        # Layer normalization helps stabilize training by normalizing the inputs to have mean 0 and variance 1.
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # 1. Attention Phase
        # We apply LayerNorm BEFORE attention (Pre-Norm formulation).
        # We add the input 'x' to the output (Residual Connection) to help gradients flow towards the input.
        x = x + self.sa(self.ln1(x))
        
        # 2. Feed-Forward Phase
        # Similarly, LayerNorm before FF, and add residual connection.
        x = x + self.ffwd(self.ln2(x))
        return x

# =============================================================================
# 5. The Tiny GPT Model
# =============================================================================
class TinyGPT(nn.Module):
    """
    The main GPT (Generative Pre-trained Transformer) model architecture.
    
    This model takes indices (tokens) as input and predicts the logits for the next token in the sequence.
    """
    def __init__(self, vocab_size, embedding_dim, block_size, n_heads, n_layers, dropout):
        super().__init__()
        # Token embeddings: look up the vector for each token (content information)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embeddings: learn a vector for each position in the sequence (positional information)
        # Transformers have no inherent notion of order, so we must inject it.
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads, dropout) for _ in range(n_layers)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        # Final linear layer (Language Model Head)
        # Projects the final hidden states to the vocabulary size to get logits for each word.
        self.head = nn.Linear(embedding_dim, vocab_size) 
        
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape # Batch size, Sequence length
        
        # 1. Get Embeddings
        # Token embeddings
        tok_emb = self.token_embedding(idx) # (B, T, embedding_dim)
        
        # Position embeddings
        # We create a simple range [0, 1, ..., T-1] and look up their embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, embedding_dim)
        
        # Combine token and position embeddings
        x = tok_emb + pos_emb  # Broadcasting adds pos_emb to every batch element
        
        # 2. Pass through Transformer Blocks
        x = self.blocks(x) 
        x = self.ln_f(x)
        
        # 3. Calculate Logits
        logits = self.head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # If we are training, we also compute the loss.
            # We need to reshape logits and targets for the cross_entropy function.
            B, T, C = logits.shape 
            # Flatten the batch and time dimensions: (B*T, C)
            logits_reshaped = logits.view(B*T, C)
            # Flatten the targets: (B*T)
            targets_reshaped = targets.view(B*T)
            
            # Calculate Cross-Entropy Loss
            # This measures how well the logits predict the correct next token.
            loss = F.cross_entropy(logits_reshaped, targets_reshaped) 
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates new text given a starting context `idx`.
        
        Args:
            idx (torch.Tensor): The starting context of indices (B, T).
            max_new_tokens (int): The number of new tokens to generate.
            
        Returns:
            torch.Tensor: The sequence with generated tokens appended (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop the context to the last `block_size` tokens.
            # The model was trained with a fixed context window (block_size), so we can't feed it more than that.
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on the last time step (the prediction for the next token)
            logits = logits[:, -1, :] # (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # Sample the next token from the distribution
            next_idx = torch.multinomial(probs, 1) # (B, 1)
            
            # Append the sampled index to the running sequence
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        return idx
