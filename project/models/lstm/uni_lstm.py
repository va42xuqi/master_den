import torch.nn as nn
import torch

from project.models.one_step_model import OneStepModel
from project.models.lmu.lmu_utils.lmu_layer import LMU


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
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, statics):
        x = x.permute(0, 2, 1)
        x = self.conv(x)

        x = x.permute(0, 2, 1)
        return x


class UniLSTM(OneStepModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        use_cnn = False
        if use_cnn:
            self.encoder = CNNEncoder(
                in_channels=6,
                out_channels=self.hidden_size,
                kernel_size=26,
                stride=26,
                padding=0,
            )
        self.x_preprocessing = LMU(
            input_size=self.in_features // 2,
            hidden_size=self.hidden_size,
            memory_size=self.hidden_size,
            theta=25,
            learn_a=True,
            learn_b=True,
        )
        self.y_preprocessing = LMU(
            input_size=self.in_features // 2,
            hidden_size=self.hidden_size,
            memory_size=self.hidden_size,
            theta=25,
            learn_a=True,
            learn_b=True,
        )
        self.x_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        self.y_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        self.x_fc_out = nn.Linear(
            self.hidden_size, self.prediction_len * self.output_size // 2
        )
        self.y_fc_out = nn.Linear(
            self.hidden_size, self.prediction_len * self.output_size // 2
        )
        self.state = None

    def forward(self, src, statics):
        h0 = torch.zeros(1, src.size(0), self.hidden_size).to(src.device)
        c0 = torch.zeros(1, src.size(0), self.hidden_size).to(src.device)
        state = (h0, c0)

        x_out, y_out, x_statics, y_statics = self.preprocess_data((src, statics))

        x_out = x_out.flatten(2, 3)
        x_encod, x_state = self.x_preprocessing(x_out, self.state)
        x_src = self.x_lstm(x_encod, state)[1][1].squeeze(0)
        x_src = self.x_fc_out(x_src)

        y_out = y_out.flatten(2, 3)
        y_encod, y_state = self.y_preprocessing(y_out, self.state)
        y_src = self.y_lstm(y_encod, state)[1][1].squeeze(0)
        y_src = self.y_fc_out(y_src)

        src = torch.cat([x_src.unsqueeze(-1), y_src.unsqueeze(-1)], dim=-1)
        return src

    def preprocess_data(self, data):
        src, statics = data
        src = torch.cat([src[:, :, : self.num_players], src[:, :, -1:]], dim=2)
        statics = torch.cat([statics[:, : self.num_players], statics[:, -1:]], dim=1)

        # x and y preprocessing
        x_src = torch.cat([src[..., 0:1], src[..., 2:3]], dim=-1)
        y_src = torch.cat([src[..., 1:2], src[..., 3:4]], dim=-1)
        x_statics = statics[:, :, 0:1]
        y_statics = statics[:, :, 1:2]

        basket = (
            self.basket_features.to(src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1, 1, 1)
        )
        sign = statics[:, 0, 0] * 2 - 1
        basket[..., 2:] *= sign.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_basket = torch.cat([basket[..., 0:1], basket[..., 2:3]], dim=-1)
        y_basket = torch.cat([basket[..., 1:2], basket[..., 3:4]], dim=-1)

        x_out = torch.cat([x_src, x_basket], dim=2)
        y_out = torch.cat([y_src, y_basket], dim=2)
        return x_out, y_out, x_statics, y_statics
