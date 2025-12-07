import torch

# =============================================================================
# Hyperparameters Configuration
# =============================================================================
# This file contains all the configuration settings for the model and training process.
# Centralizing these parameters makes it easier to experiment with different settings.

# Device Configuration
# --------------------
# We check if a CUDA-enabled GPU is available. If so, we use it for faster training.
# Otherwise, we fall back to the CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Parameters
# ---------------
block_size = 8       # Context length: The maximum number of tokens the model can attend to at once.
                     # (e.g., if block_size is 8, the model uses the last 8 words to predict the next one).
batch_size = 16      # Batch size: The number of sequences processed in parallel during training.
                     # Larger batches provide more stable gradient estimates but use more memory.

# Model Parameters
# ----------------
embedding_dim = 32   # Embedding dimension: The size of the vector representation for each token.
                     # Higher dimensions can capture more complex relationships but require more compute.
n_heads = 2          # Number of attention heads: How many "perspectives" the model uses in the attention mechanism.
                     # (embedding_dim must be divisible by n_heads).
n_layers = 2         # Number of Transformer blocks: The depth of the neural network.
                     # More layers allow the model to learn more complex patterns.
dropout = 0.0        # Dropout rate: The probability of randomly zeroing out elements during training.
                     # This acts as a regularization technique to prevent overfitting.

# Training Parameters
# -------------------
learning_rate = 1e-3 # Learning rate: The step size for the optimizer (AdamW).
                     # Controls how much the model weights are updated during training.
epochs = 2000        # Number of training iterations (steps).
                     # High number of epochs ensures the model sees enough data to learn.
eval_interval = 200  # Evaluation interval: How often (in steps) to print the training loss.
