import torch
import torch.nn as nn
import lightning.pytorch as pl

from badass_trajectory_predictor.models.parallel_model import (
    ParallelModel,
    NonTargetPlayer,
)


class TargetPlayer(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super(TargetPlayer, self).__init__()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.output_size = 64
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True).to(device)

        self.h = None
        self.c = None

    def forward(self, input_data, reset_state=True):
        if reset_state:
            self.h = torch.zeros(1, input_data.size(0), self.lstm.hidden_size).to(
                input_data.device
            )
            self.c = torch.zeros(1, input_data.size(0), self.lstm.hidden_size).to(
                input_data.device
            )
        lstm_output, (self.h, self.c) = self.lstm(input_data, (self.h, self.c))
        return lstm_output[:, -1]


class ParallelLSTM(ParallelModel):
    def __init__(self, hidden_size, data):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(hidden_size=hidden_size, data=data)
        self.target = TargetPlayer(
            input_size=self.input_size + 4, hidden_size=hidden_size
        )
        self.non_target = nn.ModuleList()
        for _ in range(self.num_players - 1):
            non_target_module = NonTargetPlayer(
                input_size=self.input_size + 2, hidden_size=hidden_size
            )
            self.add_module(f"non_target_{_}", non_target_module)
            self.non_target.append(non_target_module)
        self.concatenation_layer = nn.Linear(
            self.num_players * hidden_size,
            out_features=self.output_size * self.num_players,
        ).to(device)

    def forward(self, input_data, start_pos, reset_state=True):
        input_features = self.target_function(input_data, start_pos)
        target_features = input_features[0]
        non_target_features = input_features[1:]

        lstm_outputs = []
        target_input = input_data[:, 0, :, :]
        target_input = torch.cat(
            (target_input, target_features[:, 0].swapaxes(1, 2)), dim=2
        )
        lstm_output = self.target(target_input, reset_state=reset_state)
        lstm_outputs.append(lstm_output)

        for i in range(1, self.num_players):
            player_input = input_data[:, i, :, :]
            player_input = torch.cat(
                (player_input, non_target_features[i - 1][:, 0].swapaxes(1, 2)), dim=2
            )
            lstm_output = self.non_target[i - 1](player_input, reset_state=reset_state)
            lstm_outputs.append(lstm_output)

        concatenated_output = torch.cat(lstm_outputs, dim=1)
        output = self.concatenation_layer(concatenated_output)

        return output
