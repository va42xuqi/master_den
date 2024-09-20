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

class OneStepLSTM(OneStepModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm1 = nn.LSTM(
            input_size=self.in_features,
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
