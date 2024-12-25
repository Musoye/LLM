import torch
from datasets.train import calc_loss_loader, calc_loss_batch, evaluate_model, generate_and_print_sample, val_loader, train_loader
from model.model import GPTModel
import tiktoken

GPT_CONFIG_124M = {
 "vocab_size": 50257, # Vocabulary size
 "context_length": 256, # Context length
 "emb_dim": 768, # Embedding dimension
 "n_heads": 12, # Number of attention heads
 "n_layers": 12, # Number of layers
 "drop_rate": 0.1, # Dropout rate
 "qkv_bias": False # Query-Key-Value bias
}


def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], [] 
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs): 
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() 
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            loss.backward() 
            optimizer.step() 
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
                )
            generate_and_print_sample( 
                model, tokenizer, device, start_context
                )
    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    with torch.no_grad(): 
        train_loss = calc_loss_loader(train_loader, model, device) 
        val_loss = calc_loss_loader(val_loader, model, device)
    optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.0004, weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),}, 
        "model_and_optimizer.pth"
        )