import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.functional import scaled_dot_attention, get_positions, get_sinusoid_table


class ScaledDotProductAttention(nn.Module):
    r"""
    Inputs: Q, K, V
        - **Q** of shape `(batch, len1, d_k)`
        - **K** of shape `(batch, len2, d_k)`
        - **V** of shape `(batch, len2, d_k)`
    Outputs: output
        - **output** of shape `(batch, len1, d_k)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        return scaled_dot_attention(Q, K, V)


class MultiHeadAttention(nn.Module):
    r"""
    In the original paper, it is required that `input_size = num_heads * d_v`
    Args:
        num_heads: number of parallel projections
        input_size: dimension of model
        d_k: dimension of key
        d_v: dimension of value
    Inputs: V, K, Q
        - **V** of shape `(batch, len1, input_size)`
        - **K** of shape `(batch, len1, input_size)`
        - **Q** of shape `(batch, len2, input_size)`
        - **mask** of shape `(batch, len2, len1)`
    Outputs: output
        - **output** of shpae `(batch, len2, input_size)`
    """

    def __init__(self, num_heads, input_size, d_k, d_v):

        super().__init__()
        h = num_heads
        self.num_heads = h
        self.d_k = d_k
        self.d_v = d_v

        self.v_proj = nn.Linear(input_size, h * d_v)
        self.k_proj = nn.Linear(input_size, h * d_k)
        self.q_proj = nn.Linear(input_size, h * d_k)

        self.o_proj = nn.Linear(h * d_v, input_size)

    def forward(self, V, K, Q, mask=None):

        h, d_k, d_v = self.num_heads, self.d_k, self.d_v

        batch, len1, _ = V.size()
        batch, len2, _ = Q.size()

        v = self.v_proj(V).view(batch, len1, h, d_v)
        k = self.k_proj(K).view(batch, len1, h, d_k)
        q = self.q_proj(Q).view(batch, len2, h, d_k)

        v = v.permute(2, 0, 1, 3).contiguous().view(h * batch, len1, d_v)
        k = k.permute(2, 0, 1, 3).contiguous().view(h * batch, len1, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(h * batch, len2, d_k)

        if mask is not None:
            mask = mask.repeat(h, 1, 1)
        output = scaled_dot_attention(q, k, v, mask=mask)

        output = output.view(h, batch, len2, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            batch, len2, h * d_v)

        output = self.o_proj(output)
        return output


class PositionalEncoder(nn.Module):

    def __init__(self, max_len, embedding_dim):
        super().__init__()
        n_positions = max_len + 1
        weights = get_sinusoid_table(n_positions, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(weights)

    def forward(self, x):
        pos = get_positions(x)
        embeded = self.embedding(pos)
        return embeded


class EncoderLayer(nn.Module):
    r"""
    Inputs: x
        - **x** of shape `(batch, seq_len, d_model)`
    Outputs: out
        - **out** of shape `(batch, seq_len, d_model)`
    """

    def __init__(self, d_model, num_heads, dropout=0.1):

        super().__init__()
        n_k = n_v = d_model / num_heads
        self.mh1 = MultiHeadAttention(h, d_model, n_k, n_v)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.mh1(x, x, x)
        x = self.dropout1(x)
        x = self.ln1(x + residual)

        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout2(x)
        x = self.ln2(x + residual)
        return x


class DecoderLayer(nn.Module):
    r"""
    Inputs: x, V, K
        - **x** of shape `(batch, len1, d_model)`
        - **V** of shape `(batch, len2, d_model)`
        - **K** of shape `(batch, len2, d_model)`
        - **mask** of shape `(batch, len1, len1)`
    Outputs: out
        - **out** of shape `(batch, len1, d_model)`
    """

    def __init__(self, d_model, num_heads, dropout=0.1):

        super().__init__()
        d_k = d_v = d_model / num_heads
        self.mh1 = MultiHeadAttention(h, d_model, d_k, d_v)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.mh2 = MultiHeadAttention(h, d_model, d_k, d_v)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, V, K, mask):

        residual = x
        x = self.mh1(x, x, x, mask)
        x = self.dropout1(x)
        x = self.ln1(x + residual)

        residual = x
        x = self.mh1(V, K, x)
        x = self.dropout2(x)
        x = self.ln2(x + residual)

        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout3(x)
        x = self.ln3(x + residual)
        return x
