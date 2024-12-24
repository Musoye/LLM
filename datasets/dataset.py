#!/usr/bin/python3
import torch
from torch.utils.data import DataLoader, Dataset
#from tokenizer.simple_tokenizer import tokenizer
from tokenizer.extract import raw_text

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2") ##BPE -> byte pair encoding(Implemented by openai)
# text = "Akwirw ier"
# ch = tokenizer.encode(text)
# for ele in ch:
#     print(tokenizer.decode([ele]))
# print(tokenizer.decode(ch)) 

# print(len(tokenizer.encode(raw_text)))

#integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride): 
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, ids):
        return (self.input_ids[ids], self.target_ids[ids])

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") 
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) 
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
        )
    return dataloader

if __name__ == "__main__":
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader_v1( raw_text, batch_size=8, max_length=max_length,
                                    stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

