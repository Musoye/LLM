import torch
import torch.nn as nn

class CasualAttention(nn.Module):

    def __init__(self, d_in, d_out, dropout, context_length, typ_base=False):
        super().__init__()
        self.d_out = d_out
        self.w_key = nn.Linear(d_in, d_out, bias=typ_base)
        self.w_value = nn.Linear(d_in, d_out, bias=typ_base)
        self.w_query = nn.Linear(d_in, d_out, bias=typ_base)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_key(x)
        queries = self.query(x)
        values = self.values(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, dropout, context_length, num_heads, typ_base=False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, dropout, context_length,
                                                    num_heads, typ_base)
                    for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        