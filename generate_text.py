import torch
from hyperparameters import *
from data_handling import load_data
from model_architecture import TinyGPT
import os

# =============================================================================
# Text Generation Script
# =============================================================================
# This script loads a trained model and generates new text based on a starting word.
# 1. Sets up the environment.
# 2. rebuilds the model structure.
# 3. Loads the saved weights (state_dict).
# 4. Encodes a starting word, feeds it to the model, and decodes the output.

# =============================================================================
# 1. Setup
# =============================================================================
print(f"Using device: {device}")

# Load data (needed for vocabulary and mappings)
# Even though we aren't training, we need the vocab_size to build the model
# and the word2idx/idx2word mappings to encode input and decode output.
_, vocab_size, word2idx, idx2word = load_data()

# Initialize Model
# We must create the exact same model architecture as we used for training.
model = TinyGPT(vocab_size, embedding_dim, block_size, n_heads, n_layers, dropout)
model = model.to(device)

# Load trained weights
model_path = 'tiny_gpt_model.pth'
if os.path.exists(model_path):
    # map_location ensures we can load a GPU-trained model on CPU if needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
else:
    print("Model file not found. Please run train_model.py first.")
    exit()

model.eval() # Set to evaluation mode (disables dropout, etc.)

# =============================================================================
# 2. Generate Text
# =============================================================================
print("\n--- Generating Text ---")

start_word = "artificial"
if start_word in word2idx:
    # Convert start word to tensor
    # We create a batch of size 1 containing the index of the start word.
    context = torch.tensor([[word2idx[start_word]]], dtype=torch.long, device=device)
    
    # Generate new tokens
    # The model will append new predicted tokens to the context.
    out = model.generate(context, max_new_tokens=15)
    
    # Decode to words
    # Convert the tensor of indices back to a readable string.
    generated_text = " ".join(idx2word[int(i)] for i in out[0])
    print(f"Input: {start_word}")
    print(f"Generated: {generated_text}")
else:
    print(f"Start word '{start_word}' not in vocabulary.")
