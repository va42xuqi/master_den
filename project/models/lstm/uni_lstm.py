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

    def forward(self, x, statics):
        x = x.permute(0, 2, 1)
        x = self.conv(x)

        x = x.permute(0, 2, 1)
        return x


class UniLSTM(OneStepModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm1x = nn.LSTM(
            input_size=self.in_features // 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        self.dropout_layer = nn.Dropout(0.1)
        self.lstm2x = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        self.lstm1y = nn.LSTM(
            input_size=self.in_features // 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        self.lstm2y= nn.LSTM(
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
        state1x = (h0.clone(), c0.clone())
        state2x = (h0.clone(), c0.clone())
        state1y = (h0.clone(), c0.clone())
        state2y = (h0.clone(), c0.clone())

        x_out, y_out, _, _ = self.preprocess_data((src, statics))

        x_out = x_out.flatten(2, 3)
        x_out, _ = self.lstm1x(x_out, state1x)
        x_out = self.dropout_layer(x_out)
        x_out, _ = self.lstm2x(x_out, state2x)
        x_out = self.x_fc_out(x_out[:, -1:])

        y_out = y_out.flatten(2, 3)
        y_out, _ = self.lstm1y(y_out, state1y)
        y_out = self.dropout_layer(y_out)
        y_out, _ = self.lstm2y(y_out, state2y)
        y_out = self.y_fc_out(y_out[:, -1:])

        x_out = torch.swapaxes(x_out, 1, 2)
        y_out = torch.swapaxes(y_out, 1, 2)
        src = torch.cat([x_out, y_out], dim=-1)
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
