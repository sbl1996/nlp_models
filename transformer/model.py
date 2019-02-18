import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.layers import MultiHeadAttention, PositionalEncoder
from transformer.functional import get_subsequent_mask


class EncoderLayer(nn.Module):
    r"""
    Inputs: x
        - **x** of shape `(batch, seq_len, d_model)`
    Outputs: out
        - **out** of shape `(batch, seq_len, d_model)`
    """

    def __init__(self, d_model, num_heads, dropout=0.1):

        super().__init__()
        n_k = n_v = d_model // num_heads
        self.mh1 = MultiHeadAttention(num_heads, d_model, n_k, n_v)
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
        d_k = d_v = d_model // num_heads
        self.mh1 = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.mh2 = MultiHeadAttention(num_heads, d_model, d_k, d_v)
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


class Encoder(nn.Module):
    def __init__(
            self,
            input_size, d_model, max_len,
            num_layers, num_heads,
            dropout=0.1):

        super().__init__()
        self.embedding = nn.Embedding(input_size, d_model)
        self.pe = PositionalEncoder(max_len, d_model)
        self.layer_stack = nn.Sequential(*[
            EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x) + self.pe(x)
        x = self.layer_stack(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            output_size, d_model, max_len,
            num_layers, num_heads,
            dropout=0.1):

        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, d_model)
        self.pe = PositionalEncoder(max_len, d_model)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, target, encoder_outputs):
        mask = get_subsequent_mask(target)

        x = self.embedding(target) + self.pe(target)
        for i in range(self.num_layers):
            x = self.layer_stack[i](x, encoder_outputs, encoder_outputs, mask)
        return x


class Transformer(nn.Module):
    r"""
    Args:
        input_size: 
        input_max_len:
        output_size:
        output_max_len:
        d_model:
        num_heads:
        encoder_layers:
        decoder_layers:
    """

    def __init__(
            self,
            input_size, input_max_len, output_size, output_max_len,
            d_model, num_heads, encoder_layers, decoder_layers):
        super().__init__()
        self.encoder = Encoder(input_size, d_model,
                               input_max_len, encoder_layers, num_heads)
        self.decoder = Decoder(output_size, d_model,
                               output_max_len, decoder_layers, num_heads)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, input, target):
        encoder_outputs = self.encoder(input)
        outputs = self.decoder(target, encoder_outputs)
        outputs = self.fc(outputs)
        outputs.transpose_(1, 2)
        return F.log_softmax(outputs, dim=1)
