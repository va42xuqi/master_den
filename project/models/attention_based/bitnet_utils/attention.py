__author__ = "Jonathan Ott"

import torch
import torch.nn as nn
from .bitlinear import BitLinear


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_out,
        n_heads,
        dropout=0.0,
        # Flash attention enabled by default
        attention_backend=None,
    ):
        super().__init__()

        if attention_backend is None:
            attention_backend = {
                "enable_math": True,
                "enable_flash": False,
                "enable_mem_efficient": False,
            }

        self.n_heads = n_heads
        self.n_hiddens = n_out

        # Dimensions heads
        self.d_k = n_out // n_heads

        self.dropout = dropout

        self.q_linear = BitLinear(n_out, n_out)
        self.k_linear = BitLinear(n_out, n_out)
        self.v_linear = BitLinear(n_out, n_out)

        self.out = BitLinear(n_out, n_out)

        self.attention_backend = attention_backend

    def forward(self, queries, keys, values, mask=None):

        # Compute subspace representations
        queries = self.q_linear(queries)
        keys = self.k_linear(keys)
        values = self.v_linear(values)

        # Split into heads
        queries = self.split_into_heads(queries)
        keys = self.split_into_heads(keys)
        values = self.split_into_heads(values)

        if mask is not None:
            mask = mask.view(-1, 1, *mask.shape[-2:])

            # Scaled dot product attention
        with torch.backends.cuda.sdp_kernel(**self.attention_backend):
            # x = self.attention(queries, keys, values, mask)
            # TODO: Add mask
            x = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, mask, self.dropout
            )

        # Concat
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        return self.out(x)

    def split_into_heads(self, x):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.d_k)
        return x.transpose(1, 2)


if __name__ == "__main__":

    multi_head_attention = MultiHeadAttention(n_out=100, n_heads=5)

    X = torch.ones((2, 4, 100))
    Y = torch.ones((2, 6, 100))

    print(multi_head_attention(X, Y, Y))
