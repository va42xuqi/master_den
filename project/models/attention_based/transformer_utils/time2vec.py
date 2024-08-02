"""
Implementation based on: S. M. Kazemi et al., “Time2vec: Learning a vector representation of time," 2019.
"""

import torch.nn as nn
import torch


class Time2Vec(nn.Module):
    def __init__(self, out_features, activation="sin"):
        super(Time2Vec, self).__init__()

        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(1, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))

        if activation == "sin":
            self.activation = torch.sin
        elif activation == "cos":
            self.activation = torch.cos

    def forward(self, X, t):
        # Periodic activation
        periodic_components = self.activation(
            torch.matmul(t.unsqueeze(2), self.w) + self.b
        )
        # Linear portion
        linear_component = torch.matmul(t.unsqueeze(2), self.w0) + self.b0
        return torch.dstack([X, linear_component, periodic_components])


class PositionalEmbedding(nn.Module):
    def __init__(self, in_features, out_features, activation="sin", dropout=0.1):
        super(PositionalEmbedding, self).__init__()

        self.t2v = Time2Vec(out_features - (in_features - 1), activation)

        self.dropout = nn.Dropout(dropout)

        self._embedding = None

    @property
    def embedding(self):
        return self._embedding.detach().numpy()

    def forward(self, x):

        pos_embedding = self.t2v(x[..., :1])
        self._embedding = torch.cat([pos_embedding, x[..., 1:]], dim=-1)
        return self.dropout(self._embedding)


if __name__ == "__main__":

    t2v = PositionalEmbedding(2, 64)

    x = torch.rand(1000, 120, 2)
    y = t2v(x)
    print(y.shape)
