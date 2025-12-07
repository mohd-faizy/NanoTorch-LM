import torch
from dataclasses import dataclass

# =============================================================================
# Hyperparameters Configuration
# =============================================================================
# This file contains all the configuration settings for the model and training process.
# Centralizing these parameters makes it easier to experiment with different settings.

@dataclass
class ModelConfig:
    """
    Configuration parameters for the NanoTorch-LM model and training loop.
    """
    # Device Configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Parameters
    block_size: int = 8       # Context length: max tokens to attend to
    batch_size: int = 16      # Number of sequences processed in parallel

    # Model Parameters
    vocab_size: int = 0       # Will be set dynamically after loading data
    embedding_dim: int = 32   # Dimension of embedding vectors
    n_heads: int = 2          # Number of attention heads
    n_layers: int = 2         # Number of Transformer blocks
    dropout: float = 0.0      # Dropout rate

    # Training Parameters
    learning_rate: float = 1e-3
    epochs: int = 2000
    eval_interval: int = 200

# Create a default configuration instance for easy import
# Note: vocab_size needs to be updated after data loading
default_config = ModelConfig()
