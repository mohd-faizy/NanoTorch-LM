import torch
import logging
import sys
import os
from hyperparameters import default_config
from data_handling import load_data
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
    Main text generation function.
    """
    # 1. Setup
    # -------------------------------------------------------------------------
    config = default_config
    logger.info(f"Using device: {config.device}")

    # Load data (needed for vocabulary and mappings)
    _, vocab_size, word2idx, idx2word = load_data()
    
    # Update config with dynamic vocabulary size
    config.vocab_size = vocab_size

    # Initialize Model
    model = TinyGPT(config)
    model = model.to(config.device)

    # Load trained weights
    model_path = 'tiny_gpt_model.pth'
    if not os.path.exists(model_path):
        logger.error("Model file not found. Please run train_model.py first.")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.exception(f"Failed to load model architecture: {e}")
        sys.exit(1)

    model.eval() 

    # 2. Generate Text
    # -------------------------------------------------------------------------
    logger.info("--- Generating Text ---")

    start_word = "artificial"
    if start_word in word2idx:
        # Convert start word to tensor
        context = torch.tensor([[word2idx[start_word]]], dtype=torch.long, device=config.device)
        
        # Generate new tokens
        with torch.no_grad():
            out = model.generate(context, max_new_tokens=15)
        
        # Decode to words
        generated_text = " ".join(idx2word[int(i)] for i in out[0])
        logger.info(f"Input: {start_word}")
        logger.info(f"Generated: {generated_text}")
    else:
        logger.error(f"Start word '{start_word}' not in vocabulary.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during text generation.")
        sys.exit(1)
