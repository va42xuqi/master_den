import torch.nn as nn
import torch
import numpy as np

from .bitlinear import BitLinear


def positional_encoding(max_len, d_model):
    """
    Generates a positional encoding matrix for a given maximum sequence length and model dimension.
    """
    pos_enc = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                pos_enc[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    return pos_enc


class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dims, output_dims, sequence_length, dtype=torch.float32):

        super().__init__()

        self.pos_enc = positional_encoding(sequence_length, output_dims)
        self.pos_enc = torch.as_tensor(self.pos_enc, dtype=dtype)

        self.linear = BitLinear(input_dims, output_dims)

    def forward(self, x):

        emb = self.linear(x)

        return emb + self.pos_enc.to(x.device)[: emb.shape[-2]]
