__author__ = "Jonathan Ott"

from typing import Optional

import torch
import torch.nn as nn


class VariableSelection(nn.Module):
    def __init__(self, input_size, hidden_size, c_size=0):
        super().__init__()
        self.input_size = input_size  # Define input size
        self.hidden_size = hidden_size  # Define hidden size

        self.flattened_grn = GRN(
            hidden_size=hidden_size * input_size,
            input_size=hidden_size * input_size,
            c_size=c_size,
            output_size=input_size,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.features_grn = nn.ModuleList([GRN(hidden_size) for _ in range(input_size)])

        self.features_linear = nn.Linear(1, hidden_size)

    def forward(self, x, c: Optional[torch.Tensor] = None):
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        eta = self.features_linear(x.unsqueeze(-1))

        flattened = eta.view(-1, x.size(1), self.hidden_size * self.input_size)
        v = self.softmax(self.flattened_grn(flattened, c))

        etas = []
        for i in range(self.input_size):
            tmp = self.features_grn[i](eta[:, :, i])
            etas.append(tmp)

        out = torch.stack(etas, dim=-2)
        out = torch.sum(out * v.unsqueeze(-1), dim=-2)

        if is_2d:
            out = out.squeeze(1)

        return out


class GLU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = self.linear1(x)
        tmp2 = self.linear2(x)
        out = self.sigmoid(tmp) * tmp2

        return out


class GRN(nn.Module):
    def __init__(self, hidden_size, input_size=None, c_size=None, output_size=None):
        super().__init__()

        input_size = input_size if input_size is not None else hidden_size

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(input_size, hidden_size)
        if c_size:
            self.linear3 = nn.Linear(c_size, hidden_size, bias=False)
        if output_size:
            self.linear4 = nn.Linear(hidden_size, output_size)

        self.out = False if output_size is None else True

        output_size = output_size if output_size is not None else hidden_size
        self.glu = GLU(hidden_size, output_size)
        self.ln = nn.LayerNorm(output_size)
        self.elu = nn.ELU()

    def forward(self, a, c: Optional[torch.Tensor] = None):
        if c is None:
            tmp2 = 0
        else:
            tmp2 = self.linear3(c).unsqueeze(1)
        tmp = self.linear2(a) + tmp2
        n2 = self.elu(tmp)
        n1 = self.linear1(n2)

        tmp = self.glu(n1)

        if self.out:
            a = self.linear4(a)
        out = self.ln(a + tmp)

        return out


class AddNorm(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(size)

    def forward(self, x, y):
        return self.norm(self.dropout(y) + x)


class PositionwiseFFN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout=0.2, activation=nn.GELU()):
        super().__init__()
        self.poswise_linear1 = nn.Linear(n_input, n_hidden)
        self.poswise_linear2 = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.poswise_linear1(x))
        return self.poswise_linear2(self.dropout(x))


def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def masked_softmax(X, mask=None, value=-1e9):
    """Perform softmax operation by masking elements on the last axis"""

    X = X.masked_fill(mask == 0, float("-inf"))

    softmax_output = nn.functional.softmax(X, dim=-1)

    return softmax_output * mask


def padding_mask(len_queries, len_keys, valid_lens, device="cpu"):
    """Create mask to reduce squence length"""
    valid_lens = torch.as_tensor(valid_lens, device=device)
    batch_size = len(valid_lens)

    mask = torch.arange(len_keys, dtype=torch.float32, device=device)
    mask = (
        torch.ones((batch_size, len_queries, len_keys), device=device)
        * mask[None, None, :]
    )

    # Expand dims such that np broadcasting with mask is possible
    valid_lens = valid_lens[:, None, None]

    # Multihead-attention masks out true values
    mask = mask > valid_lens

    # Multihead attention expects binary mask
    return mask


def decoder_padding(x, value):
    tgt_in = nn.functional.pad(x, (0, 0, 1, 0), "constant", value)
    tgt_out = nn.functional.pad(x, (0, 0, 0, 1), "constant", value)
    return tgt_in, tgt_out


if __name__ == "__main__":
    input_size = 4
    hidden_size = 128
    batch_size = 16
    sequence_length = 20

    vs = VariableSelection(input_size, hidden_size, c_size=10)
    x = torch.rand(batch_size, sequence_length, input_size)
    c = torch.rand(batch_size, 10)
    out = vs(x, c)
    print(out.shape)
