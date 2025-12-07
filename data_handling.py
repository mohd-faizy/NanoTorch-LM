import torch
from typing import Tuple, Dict

# =============================================================================
# Data Handling
# =============================================================================
# This module handles data preparation, including:
# 1. Creating the small training corpus.
# 2. Building the vocabulary (mapping between words and integers).
# 3. Encoding the text into tensors.

def load_data() -> Tuple[torch.Tensor, int, Dict[str, int], Dict[int, str]]:
    """
    Prepares the dataset for training.
    
    This function:
    - Defines a small list of sentences as the corpus.
    - Appends an <END> token to each sentence to mark boundaries.
    - builds a vocabulary of unique words.
    - Creates mappings: Word -> Integer (word2idx) and Integer -> Word (idx2word).
    - Encodes the entire corpus into a single sequence of integers.

    Returns:
        data (torch.Tensor): The entire dataset encoded as indices.
        vocab_size (int): The number of unique tokens in the vocabulary.
        word2idx (dict): Mapping from word string to integer index.
        idx2word (dict): Mapping from integer index to word string.
    """
    # 1. The Corpus (Training Data)
    # A list of simple sentences for the model to learn from.
    corpus = [
        "artificial intelligence is transforming the world",
        "machine learning models learn from data",
        "deep learning uses neural networks",
        "python is a popular programming language",
        "pytorch makes building models easy",
        "coding is a valuable skill to have",
        "the sun rises in the east every day",
        "reading books expands your knowledge",
        "practice makes a man perfect",
        "consistency is the key to success"
    ]
    
    # Add <END> token to mark the end of sentences.
    # This helps the model learn when a sentence is finished.
    corpus = [s + " <END>" for s in corpus]
    
    # Combine all sentences into one long string.
    text = " ".join(corpus)
    
    # 2. Build Vocabulary
    # Find all unique words in the text to create the vocabulary.
    words = sorted(list(set(text.split())))
    vocab_size = len(words)
    
    # 3. Create Mappings
    # word2idx: converts a human-readable word to a unique integer ID.
    word2idx = {w: i for i, w in enumerate(words)}
    # idx2word: converts an integer ID back to the human-readable word.
    idx2word = {i: w for w, i in word2idx.items()}
    
    # 4. Encode Data
    # Convert the entire text string into a PyTorch tensor of integers.
    # This tensor is what we will feed into the model.
    data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
    
    return data, vocab_size, word2idx, idx2word

def get_batch(
    data: torch.Tensor, 
    batch_size: int, 
    block_size: int, 
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random batch of data for training.
    
    Args:
        data (torch.Tensor): The entire encoded text tensor.
        batch_size (int): Number of sequences in the batch.
        block_size (int): Context length for each sequence.
        device (str): Device to move the batch to ('cpu' or 'cuda').
        
    Returns:
        x (torch.Tensor): Inputs (context) of shape (batch_size, block_size).
        y (torch.Tensor): Targets (next word) of shape (batch_size, block_size).
    """
    # Ensure dataset is large enough
    if len(data) <= block_size:
         raise ValueError("Dataset is too small for the requested block_size.")

    # Choose random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input (x) and target (y) sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to the configured device
    x, y = x.to(device), y.to(device)
    return x, y
