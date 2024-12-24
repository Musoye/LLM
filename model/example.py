#!/usr/bin/python3
GPT_CONFIG_124M = {
 "vocab_size": 50257, # Vocabulary size
 "context_length": 256, # Context length
 "emb_dim": 768, # Embedding dimension
 "n_heads": 12, # Number of attention heads
 "n_layers": 12, # Number of layers
 "drop_rate": 0.1, # Dropout rate
 "qkv_bias": False # Query-Key-Value bias
}

from model.model import GPTModel
import torch

torch.manual_seed(32)
model = GPTModel(GPT_CONFIG_124M)


if __name__ == "__main__":
    #print(model)
    parameters = sum([p.numel() for p in model.parameters()])
    print(f"GPT 2 has {parameters} parameters")