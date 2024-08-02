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


class OneStepLSTM(OneStepModel):
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
        self.preprocessing = LMU(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            memory_size=self.hidden_size,
            theta=25,
            learn_a=True,
            learn_b=True,
        )
        self.dropout_layer = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
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
        encod, state = self.preprocessing(out, self.state)
        h0 = torch.zeros(1, src.size(0), self.hidden_size).to(src.device)
        c0 = torch.zeros(1, src.size(0), self.hidden_size).to(src.device)
        state = (h0, c0)
        src = self.lstm(encod, state)[1][1].squeeze(0)
        src = self.fc_out(src)
        return src.view(src.size(0), self.prediction_len, self.output_size)
