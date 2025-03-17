import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.getcwd())) 
sys.path.insert(0, root_dir)
from config import SEQ_LEN

class PositionalEmbedding(nn.Module):
    def __init__(self, dim_model, max_len, n=10000):
        super().__init__()
        pos_emb = torch.zeros(max_len, dim_model).float()
        pos_emb.requires_grad = False

        for pos in range(max_len):
            for i in range(0, dim_model, 2):
                pos_emb[pos, i] = np.sin(pos/(n**(i/dim_model)))
                pos_emb[pos, i+1] = np.cos(pos/(n**(i/dim_model)))
        # self.pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer("pos_emb", pos_emb.unsqueeze(0))

    def forward(self, x):
        # in case a different seq_len is used during inference
        return self.pos_emb[:, :x.shape[1], :]  
        # return self.pos_emb


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len=SEQ_LEN, dropout=0.1, debug=False):
        super().__init__()
        self.embed_size = embed_size
        self.debug = debug
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(dim_model=embed_size, max_len=seq_len)
        self.dropout = nn.Dropout(p=dropout)
    
    
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.segment(segment_label) + self.position(sequence)
        x = self.dropout(x)
        if self.debug:
            return x, self.segment(segment_label)   
        else:
            return x
        


class SelfAttention(nn.Module):
    def __init__(self, embed_size, n_heads):
        super().__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        assert (self.head_dim * n_heads == embed_size), "embed_size needs to be divisable by the number of heads" 

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)


    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # (batch_size, seq_len, n_heads, head_dim)
        values = values.reshape(N, value_len, self.n_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.n_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.n_heads, self.head_dim)
                                                                            # Scaled Dot-Product Attention:
        scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])           # MatMul Q and K
        scores = scores.masked_fill(mask == 0, -1e9)                        # Mask
        attention = torch.softmax(scores/(self.embed_size**(1/2)), dim=3)   # Scale + Softmax
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])          # MatMul attention and V

                                                                            # Multi-Head Attention
        out = out.reshape(N, query_len, self.head_dim*self.n_heads)         # Concat
        out = self.fc_out(out)                                              # Linear
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, n_heads, dropout, forward_expansion=4):
        super().__init__()
        self.attention = SelfAttention(embed_size, n_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)  # Multi-Head Attention
        x = self.dropout(self.norm1(attention + query))      # Add & Norm
        ff_output = self.feed_forward(x)                     # Feed Forward
        out = self.dropout(self.norm2(ff_output + x))        # Add & Norm
        return out
    

        