import torch
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

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def generate_text_simple(model, idx, 
    max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        with torch.no_grad():
            logits = model(idx_cond)
    
    logits = logits[:, -1, :] 
    probas = torch.softmax(logits, dim=-1) 
    idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
    idx = torch.cat((idx, idx_next), dim=1)

if __name__ == "__main__":
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    print("encoded_tensor.shape:", encoded_tensor.shape)
    model.eval() 
    out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))