import torch
from hyperparameters import *
from data_handling import load_data, get_batch
from model_architecture import TinyGPT

# =============================================================================
# Training Script
# =============================================================================
# This script orchestrates the training process:
# 1. Sets up the device and loads the data.
# 2. Initializes the model and optimizer.
# 3. Runs the training loop (forward pass, loss calculation, backward pass).
# 4. Saves the trained model.

# =============================================================================
# 1. Setup
# =============================================================================
print(f"Using device: {device}")

# Load data
# We get the encoded data and the vocabulary info.
data, vocab_size, word2idx, idx2word = load_data()
print(f"Vocabulary Size: {vocab_size}")
print(f"Total tokens in data: {len(data)}")

# Initialize Model
# We create an instance of our TinyGPT model with the configurations from hyperparameters.py
model = TinyGPT(vocab_size, embedding_dim, block_size, n_heads, n_layers, dropout)
model = model.to(device) # Move model to GPU if available

# Initialize Optimizer
# We use AdamW, a standard optimizer for Transformers.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")

# =============================================================================
# 2. Training Loop
# =============================================================================
print("\n--- Starting Training ---")
for step in range(epochs):
    # Get a batch of data
    # xb is the input context, yb is the target (next word)
    xb, yb = get_batch(data, batch_size)
    
    # Forward pass
    # Feed input to model, get predictions (logits) and loss
    logits, loss = model(xb, yb)
    
    # Backward pass and Optimization
    optimizer.zero_grad(set_to_none=True) # Reset gradients
    loss.backward()                       # Compute gradients using backpropagation
    optimizer.step()                      # Update model parameters
    
    # Print progress
    if step % eval_interval == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("--- Training Finished ---")

# Save the model
# We save the state_dict (learned weights) so we can load it later for generation.
torch.save(model.state_dict(), 'tiny_gpt_model.pth')
print("Model saved to 'tiny_gpt_model.pth'")
