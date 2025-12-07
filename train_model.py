import torch
import logging
import sys
from hyperparameters import default_config
from data_handling import load_data, get_batch
from model_architecture import TinyGPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main training function.
    """
    # 1. Setup
    # -------------------------------------------------------------------------
    config = default_config
    logger.info(f"Using device: {config.device}")

    # Load data
    data, vocab_size, word2idx, idx2word = load_data()
    
    # Update config with dynamic vocabulary size
    config.vocab_size = vocab_size
    
    logger.info(f"Vocabulary Size: {vocab_size}")
    logger.info(f"Total tokens in data: {len(data)}")

    # Initialize Model
    model = TinyGPT(config)
    model = model.to(config.device)
    
    # Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {param_count} parameters.")

    # 2. Training Loop
    # -------------------------------------------------------------------------
    logger.info("--- Starting Training ---")
    
    model.train() # Explicitly set to train mode
    
    for step in range(config.epochs):
        # Get a batch of data
        xb, yb = get_batch(data, config.batch_size, config.block_size, config.device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        if loss is not None:
            loss.backward()
        optimizer.step()
        
        # Logging
        if step % config.eval_interval == 0 and loss is not None:
             logger.info(f"Step {step}, Loss: {loss.item():.4f}")

    logger.info("--- Training Finished ---")

    # Save the model
    torch.save(model.state_dict(), 'tiny_gpt_model.pth')
    logger.info("Model saved to 'tiny_gpt_model.pth'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during training.")
        sys.exit(1)
