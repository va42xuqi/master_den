import torch
import torch.nn as nn

from badass_trajectory_predictor.models.model import Model


class LSTMModel(Model):
    def __init__(self, num_layers=2, dropout=0.2, **kwargs):
        super(LSTMModel, self).__init__(**kwargs)

        self.c = None
        self.h = None
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_shape[0] * 2,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, reset_state=True, pred_len=None):
        pred_len = pred_len if pred_len is not None else self.pred_len
        if reset_state:
            self.c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )
            self.h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )
        out, (self.h, self.c) = self.lstm(
            x.reshape(x.shape[0], x.shape[1], -1), (self.h, self.c)
        )
        out = self.linear(out)

        return out
