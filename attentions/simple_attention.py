import torch
import torch.nn as nn

class SimpleAttention(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_key = nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = nn.Parameter(torch.rand(d_in, d_out))
        self.w_query = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.w_key
        queries = x @ self.w_query
        values = x @ self.w_value
        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

## Using Linear Layer

class SimpleAttentionV2(nn.Module):

    def __init__(self, d_in, d_out, typ_base=False):
        super().__init__()
        self.w_key = nn.Linear(d_in, d_out, bias=typ_base)
        self.w_value = nn.Linear(d_in, d_out, bias=typ_base)
        self.w_query = nn.Linear(d_in, d_out, bias=typ_base)

    def forward(self, x):
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec