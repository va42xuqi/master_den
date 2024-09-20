from project.models.attention_based import TrafoLayer
import torch
import torch.nn as nn

from project.models.one_step_model import OneStepModel
import numpy as np
from scipy.special import legendre


def generate_legendre_positional_encodings(seq_len, d_model, degree):
    positions = np.linspace(-1, 1, seq_len)
    legendre_features = np.stack(
        [legendre(d)(positions) for d in range(degree + 1)], axis=1
    )

    # Repeat or truncate Legendre features to match d_model
    if d_model % (degree + 1) != 0:
        raise ValueError(
            "d_model must be a multiple of the number of Legendre features"
        )

    legendre_encodings = np.tile(legendre_features, (1, d_model // (degree + 1)))
    return torch.tensor(legendre_encodings, dtype=torch.float32)


def relative_values(x, target):
    x = x - target.unsqueeze(2)
    return x


def pos_to_basket(x, basket_positions):
    pos = x.clone()
    dist = basket_positions - pos
    return dist


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, x, statics):
        x = x.permute(0, 2, 1)
        x = self.conv(x)

        x = x.permute(0, 2, 1)
        return x


class pos_trafo(OneStepModel):
    def __init__(self, n_blocks=8, n_heads=6, ffn_hidden=1024, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(4 * 26, self.hidden_size)
        self.pos_enc = generate_legendre_positional_encodings(
            self.history_len, self.hidden_size, 15
        )
        # self.lmu = LMUFFT(
        #    input_size=13 * 4,
        #    hidden_size=self.hidden_size,
        #    memory_size=256,
        #    theta=25,
        #    seq_len=self.history_len,
        # )
        self.encoder = TrafoLayer(
            n_blocks=n_blocks,
            input_dim=self.hidden_size,
            ffn_hidden=ffn_hidden,
            n_heads=n_heads,
            dropout=self.dropout,
            pos_encoding=nn.Identity(),
            generator=nn.Identity(),
            alibi=True,
        )
        self.cnn = CNNEncoder(
            in_channels=self.in_features // 2,
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
        )
        self.fc_out = nn.Linear(
            self.hidden_size, self.prediction_len * self.output_size
        )
        if self.has_goals:
            basket = torch.tensor(self.get_goal_position())
            basket_vel = torch.zeros_like(basket)
            self.basket_features = torch.cat([basket_vel, basket], dim=-1).repeat(
                self.history_len, 1, 1
            )
        self.state = None
        self.dropout = nn.Dropout(0.5)

    def forward(self, src, statics):
        out, _ = self.preprocess_data((src, statics))
        out = out.flatten(2, 3)
        # out = self.lmu(out, self.state)[0]
        out = self.cnn(out, statics)
        out = self.encoder(out)[:, -1]

        src = self.fc_out(out)
        return src.view(src.size(0), self.prediction_len, self.output_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer

    def preprocess_data(self, data):
        src, statics = data
        src = src[..., 2:]
        src = torch.cat([src[:, :, : self.num_players], src[:, :, -1:]], dim=2)
        statics = torch.cat([statics[:, : self.num_players], statics[:, -1:]], dim=1)
        basket = (
            self.basket_features.to(src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1, 1, 1)
        )
        basket = basket[..., 2:]
        sign = statics[:, 0, 0] * 2 - 1
        basket * sign.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = torch.cat([src, basket], dim=2)
        return out, statics
