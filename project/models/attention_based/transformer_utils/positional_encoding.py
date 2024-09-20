import torch.nn as nn
import torch
import numpy as np


class SinoidalPositionalEncoding2D(nn.Module):
    def __init__(self, max_sequence_length=100, embed_dim=64, num_players=22):
        super().__init__()
        num_sequences = num_players  # Num players
        sequence_length = max_sequence_length
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_sequences * sequence_length, embed_dim),
            requires_grad=False,
        )
        pos_embed = get_2d_sincos_pos_embed(embed_dim, (num_sequences, sequence_length))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # Learn scaling
        sinoidalPosEmbedScaling = 1
        sinoidalPosEmbedTrainable = True
        if sinoidalPosEmbedTrainable:
            self.scale = nn.Parameter(torch.tensor(sinoidalPosEmbedScaling))
        else:
            self.scale = sinoidalPosEmbedScaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape() -> (batch_size, sequence_length, num_sequences, embed_dim)
        res = x + self.pos_embed * self.scale
        return res


# 2D SinCos Embed from https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: object) -> object:
    """

    Parameters
    ----------
    embed_dim
    grid

    Returns
    -------
    object

    """
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class AbsPosEncoding(nn.Module):
    def __init__(self, embedding_dims, dropout, max_len=1000):
        super().__init__()

        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embedding_dims)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dims))
        pos_enc = pos * angle_rates
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        self.pos_enc = nn.Parameter(
            torch.tensor(pos_enc, dtype=torch.float32), requires_grad=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pos_enc[: x.size(1), :].unsqueeze(0)
        return self.dropout(x)


class TimeSeriesEmbedding(nn.Module):
    def __init__(
        self, input_dims, output_dims, sequence_length, dtype=torch.float32, dropout=0.0
    ):
        super().__init__()
        self.pos_enc = AbsPosEncoding(
            output_dims, dropout=dropout, max_len=sequence_length
        ).pos_enc
        sinoidalPosEmbedScaling = 1.0
        sinoidalPosEmbedTrainable = True
        if sinoidalPosEmbedTrainable:
            self.scale = nn.Parameter(torch.tensor(sinoidalPosEmbedScaling))
        else:
            self.scale = sinoidalPosEmbedScaling
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        emb = self.linear(x)
        return emb + self.pos_enc.to(x.device)[: emb.shape[-2]] * self.scale


class TimeSeriesEmbedding2D(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        sequence_length,
        dtype=torch.float32,
        num_objects=10,
    ):

        super().__init__()

        self.pos_enc = SinoidalPositionalEncoding2D(
            max_sequence_length=sequence_length,
            embed_dim=output_dims,
            num_players=num_objects,
        )
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        emb = self.linear(x.flatten(1, 2))
        out = self.pos_enc(emb)
        return out


if __name__ == "__main__":
    # Test TimeSeriesEmbedding2D
    input_dims = 4
    output_dims = 256
    sequence_length = 100
    num_objects = 10
    x = torch.randn(16, sequence_length, num_objects, input_dims)
    pos_enc = TimeSeriesEmbedding2D(
        input_dims, output_dims, sequence_length, num_objects=num_objects
    )
    out = pos_enc(x)
    print(out.shape)  # torch.Size([1, 2200, 64])
