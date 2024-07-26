import torch
import torch.nn as nn
import lightning.pytorch as pl

from badass_trajectory_predictor.models.parallel_model import (
    ParallelModel,
    NonTargetPlayer,
)
from badass_trajectory_predictor.models.lmu.lmu_utils.lmu_layer import LMU


class TargetPlayer(pl.LightningModule):
    def __init__(
        self,
        memory_size=8,
        theta=10,
        learn_a=False,
        learn_b=False,
        input_size=4,
        hidden_size=64,
    ):
        super(TargetPlayer, self).__init__()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.output_size = 64
        self.lmu = LMU(
            input_size=self.input_size + 4,
            hidden_size=self.hidden_size,
            memory_size=memory_size,
            theta=theta,
            learn_a=learn_a,
            learn_b=learn_b,
        )

        self.state = None

    def forward(self, input_data, reset_state=True):
        if reset_state:
            self.state = None
        lstm_output, self.state = self.lmu(input_data, self.state)
        return lstm_output[:, -1]


class ParallelLMU(ParallelModel):
    def __init__(self, hidden_size, data):
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
        )

    def forward(self, input_data, start_pos, reset_state=True):
        input_features = self.target_function(input_data, start_pos)
        target_features = input_features[0]
        non_target_features = input_features[1:]

        lmu_outputs = []
        target_input = input_data[:, 0, :, :]
        target_input = torch.cat(
            (target_input, target_features[:, 0].swapaxes(1, 2)), dim=2
        )
        lmu_output = self.target(target_input, reset_state=reset_state)
        lmu_outputs.append(lmu_output)

        for i in range(1, self.num_players):
            player_input = input_data[:, i, :, :]
            player_input = torch.cat(
                (player_input, non_target_features[i - 1][:, 0].swapaxes(1, 2)), dim=2
            )
            lmu_output = self.non_target[i - 1](player_input, reset_state=reset_state)
            lmu_outputs.append(lmu_output)

        concatenated_output = torch.cat(lmu_outputs, dim=1)
        output = self.concatenation_layer(concatenated_output)

        return output
