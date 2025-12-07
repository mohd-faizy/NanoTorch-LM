# Mini GPT: A PyTorch Implementation

## 📖 Project Overview

This repository provides a modular, educational implementation of a Generative Pre-trained Transformer (GPT) model using PyTorch. Designed for clarity and extensibility, it serves as a practical resource for understanding the internal mechanics of Large Language Models (LLMs), including Self-Attention mechanisms, Transformer blocks, and autoregressive text generation.

The codebase is structured to facilitate step-by-step learning, isolating critical components such as data processing, model architecture, and training logic into distinct modules.

## 📂 Repository Structure

The project follows a modular design pattern to separate concerns and improve maintainability.

```text
.
├── hyperparameters.py      # Configuration center for model and training parameters
├── data_handling.py        # Data ingestion, tokenization, and batch generation pipeline
├── model_architecture.py   # Core GPT model definition (Transformer, Attention, FeedForward)
├── train_model.py          # Training orchestration script
├── generate_text.py        # Inference script for autoregressive text generation
├── input.txt               # (Optional) Raw text corpus for training
└── README.md               # Project documentation
```

## 🏗️ Architecture Walkthrough

To gain a comprehensive understanding of the system, we recommend reviewing the modules in the following logical sequence:

### 1. Configuration Layer (`hyperparameters.py`)
Defines the global constants and hyperparameters that control the model's capacity and training dynamics.
*   **Key Parameters**: `batch_size`, `block_size` (context window), `n_embd` (embedding dimension), `n_head`, `n_layer`.

### 2. Data Pipeline (`data_handling.py`)
Implements the ETL (Extract, Transform, Load) logic for textual data.
*   **Tokenization**: Character-level mapping (converting characters to integer indices).
*   **Batching**: Generates `(input, target)` pairs for supervised learning, ensuring efficient GPU utilization.

### 3. Core Architecture (`model_architecture.py`)
Encapsulates the mathematical definition of the GPT model.
*   **`TinyGPT`**: The main container class.
*   **`Block`**: A single Transformer block containing LayerNorm, Multi-Head Attention, and Feed-Forward networks.
*   **`MultiHeadAttention`**: The mechanism allowing the model to attend to different parts of the sequence simultaneously.

### 4. Training Engine (`train_model.py`)
Orchestrates the optimization process.
*   **Optimization**: Uses `AdamW` optimizer.
*   **Loop**: Performs forward pass, loss calculation (Cross-Entropy), backward pass, and parameter updates.
*   **Checkpointing**: Saves model weights to `tiny_gpt_model.pth`.

### 5. Inference Engine (`generate_text.py`)
Demonstrates the model's generative capabilities.
*   **Sampling**: Uses the trained model to predict tokens autoregressively.
*   **Decoding**: Converts predicted token indices back into human-readable text.

## 🚀 Getting Started

### Prerequisites
*   **Python 3.8+**
*   **Package Manager**: `uv` (recommended) or `pip`

### Installation

Initialize the environment and install dependencies:

```bash
uv sync
# OR
pip install torch numpy tqdm
```

### Usage

#### 1. Train the Model
Execute the training script to optimize the model parameters on the provided corpus.

```bash
python train_model.py
```
*Output: Training logs showing loss reduction over epochs.*

#### 2. Generate Text
Run the inference script to generate text sequences using the trained weights.

```bash
python generate_text.py
```

## 📜 License

This project is licensed under the MIT License.
