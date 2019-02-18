import math

import numpy as np
import torch


def scaled_dot_attention(Q, K, V, mask=None):
    r"""
    Inputs: Q, K, V
        - **Q** of shape `(batch, len1, n_k)`
        - **K** of shape `(batch, len2, n_k)`
        - **V** of shape `(batch, len2, n_v)`
    Outputs: output
        - **output** of shape `(batch, len1, n_v)`
    """
    n_k = K.size(1)
    attn = torch.bmm(Q, K.transpose(1, 2)).div_(math.sqrt(n_k))
    if mask is not None:
        attn.masked_fill_(mask, -np.inf)
    attn = torch.softmax(attn, dim=-1)
    return torch.bmm(attn, V)


def get_sinusoid_table(n_position, embedding_dim):
    table = torch.zeros(n_position, embedding_dim)
    exp = torch.arange(embedding_dim / 2).div_(embedding_dim / 2)
    divisor = exp.new_full(exp.size(), 10000).pow_(exp)
    for i in range(1, n_position):
        weight = torch.empty(embedding_dim)
        weight[::2] = (i / divisor).sin_()
        weight[1::2] = (i / divisor).cos_()
        table[i] = weight
    return table


def get_positions(x):
    r"""
    Inputs: x
        - **x** of shape `(*, N)`

    Outputs: pos
        - **pos** of shape `(*, N)`

    Examples::

        >>> x = torch.tensor([[3,1,0,0], [0,2,1,4], [0,4,3,0]])
        >>> get_positions(x)
        tensor([[1, 2, 0, 0],
                [1, 2, 3, 0]])
    """
    mask = (x != 0).long()
    pos = mask.cumsum(dim=-1) * mask

    # return (x != 0).cumsum(dim=-1).masked_fill_(x == 0, 0)
    return pos


def get_subsequent_mask(seq):
    r"""
    Inputs: seq
        - **seq** of shape `(batch, seq_len)

    Outputs: mask
        - **mask** of shape `(batch, seq_len, seq_len)`
    """
    batch, seq_len = seq.size()
    mask = seq.new_ones((seq_len, seq_len), dtype=torch.uint8).triu_(1)
    mask = mask.expand(batch, -1, -1)
    return mask


def get_subsequent_mask2(seq):
    r"""
    General version of `get_subsequent_mask`

    Inputs: seq
        - **seq** of shape `(*, seq_len)

    Outputs: mask
        - **mask** of shape `(*, seq_len, seq_len)`
    """
    *shape, seq_len = seq.size()
    mask = seq.new_ones((seq_len, seq_len), dtype=torch.uint8).triu_(1)
    mask = mask.expand(*shape, -1, -1)
    return mask
