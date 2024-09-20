__author__ = "Denis Gosalci"

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_out,
        n_heads,
        dropout=0.0,
        # Flash attention enabled by default
        attention_backend=None,
        use_alibi=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_hiddens = n_out

        # Dimensions heads
        self.d_k = n_out // n_heads

        self.dropout = dropout

        self.q_linear = nn.Linear(n_out, n_out)
        self.k_linear = nn.Linear(n_out, n_out)
        self.v_linear = nn.Linear(n_out, n_out)

        self.out = nn.Linear(n_out, n_out)

        # Flash attention enabled by default
        self.attention_backend = (
            (
                attention_backend
                if attention_backend is not None
                else {
                    "enable_math": True,
                    "enable_flash": False,
                    "enable_mem_efficient": False,
                }
            )
            if attention_backend is None
            else attention_backend
        )

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


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_out,
        n_heads,
        dropout=0.0,
        attention_backend=None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_hiddens = n_out

        # Dimensions heads
        self.d_k = n_out // n_heads

        self.dropout = dropout

        self.q_linear = nn.Linear(n_out, n_out)
        self.k_linear = nn.Linear(n_out, n_out)
        self.v_linear = nn.Linear(self.d_k, self.d_k)

        self.out = nn.Linear(self.d_k, n_out)

        # Flash attention enabled by default
        self.attention_backend = (
            (
                attention_backend
                if attention_backend is not None
                else {
                    "enable_math": True,
                    "enable_flash": False,
                    "enable_mem_efficient": False,
                }
            )
            if attention_backend is None
            else attention_backend
        )

    def forward(self, queries, keys, values, mask=None):

        # Compute subspace representations
        queries = self.q_linear(queries)
        keys = self.k_linear(keys)

        # Split into heads
        queries = self.split_into_heads(queries)
        keys = self.split_into_heads(keys)
        values = self.split_into_heads(values)

        for i in range(values.size(1)):
            values[:, i] = self.v_linear(values[:, i])

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
        x = torch.sum(x, dim=1)

        return self.out(x)

    def split_into_heads(self, x):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.d_k)
        return x.transpose(1, 2)


if __name__ == "__main__":
    features = 128
    n_heads = 16
    interpret_mul_head = InterpretableMultiHeadAttention(
        n_out=features, n_heads=n_heads
    )
    multi_head_attention = MultiHeadAttention(n_out=features, n_heads=n_heads)

    X = torch.ones((2, 4, features))
    Y = torch.ones((2, 6, features))

    print(multi_head_attention(X, Y, Y).shape)
    print(interpret_mul_head(X, Y, Y).shape)
