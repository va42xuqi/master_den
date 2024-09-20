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

class vel_lstm(OneStepModel):
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
        self.lstm1 = nn.LSTM(
            input_size=self.in_features // 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        self.dropout_layer = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(
            self.hidden_size, self.prediction_len * self.output_size
        )
        self.state = None

    def forward(self, src, statics):
        out, _ = self.preprocess_data((src, statics))
        out = out.flatten(2, 3)
        h0 = torch.zeros(1, src.size(0), self.hidden_size).to(src.device)
        c0 = torch.zeros(1, src.size(0), self.hidden_size).to(src.device)
        state1 = (h0.clone(), c0.clone())
        state2 = (h0.clone(), c0.clone())
        out, state = self.lstm1(out, state1)
        out = self.dropout_layer(out)
        out, state = self.lstm2(out, state2)
        out = self.fc_out(out[:, -1:])
        return out.view(out.size(0), self.prediction_len, self.output_size)

    def preprocess_data(self, data):
        src, statics = data
        src = src[..., :2]
        src = torch.cat([src[:, :, : self.num_players], src[:, :, -1:]], dim=2)
        statics = torch.cat([statics[:, : self.num_players], statics[:, -1:]], dim=1)
        basket = (
            self.basket_features.to(src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1, 1, 1)
        )
        basket = basket[..., :2]
        sign = statics[:, 0, 0] * 2 - 1
        basket * sign.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = torch.cat([src, basket], dim=2)
        return out, statics
