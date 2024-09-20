import torch
import torch.nn as nn

from project.models.attention_based.transformer_utils import (
    GRN,
    AddNorm,
)


class TemporalFusionDecoder(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()
        self.static_enrichment = GRN(hidden_size=hidden_size, c_size=hidden_size)
        # self.MHA = InterpretableMultiHeadAttention(
        #    n_out=hidden_size, n_heads=n_heads, dropout=dropout
        # )
        self.addnorm = AddNorm(hidden_size, dropout)
        self.ffn = GRN(hidden_size=hidden_size)

    def forward(self, x, static=None, mask=None):
        x = self.static_enrichment(x, static)
        tmp = self.MHA(x, x, x, mask)
        x = self.addnorm(x, tmp)
        x = self.ffn(x)
        return x


if __name__ == "__main__":
    # Test the model
    hidden_size = 128
    n_heads = 8
    dropout = 0.1
    batch_size = 16

    sequence_length = 5

    decoder = TemporalFusionDecoder(hidden_size, n_heads, dropout)
    x = torch.rand(batch_size, sequence_length, hidden_size)
    # static is the static features of the data
    static = torch.rand(batch_size, hidden_size)
    out = decoder(x, static, mask=None)
    print(out.shape)
