import torch
import torch.nn as nn

from .. import ParallelModel, NonTargetPlayer
from . import TrafoLayer
from .transformer_utils import TimeSeriesEmbedding

import lightning.pytorch as pl


class TargetPlayer(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        embedding_size,
        ffn_hidden,
        n_blocks,
        n_heads,
        dataloader,
        dropout=0.2,
    ):
        super(TargetPlayer, self).__init__()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dropout = dropout

        pos_encoding = TimeSeriesEmbedding(
            input_dims=input_size,
            output_dims=embedding_size,
            sequence_length=dataloader.dataset.min_sequence_length,
        )
        generator = nn.Linear(ffn_hidden, hidden_size)
        self.transformer = TrafoLayer(
            n_blocks=n_blocks,
            input_dim=embedding_size,
            ffn_hidden=ffn_hidden,
            n_heads=n_heads,
            dropout=dropout,
            pos_encoding=pos_encoding,
            generator=generator,
        )

    def forward(self, x, do_reset=True):
        # x shape: [batch, sequence_length, features]
        outputs = self.transformer(x)
        return outputs


class ParallelTrafo(ParallelModel):
    def __init__(self, hidden_size, data):
        super().__init__(hidden_size=hidden_size, data=data)
        self.target = TargetPlayer(
            input_size=self.input_size + 4,
            hidden_size=hidden_size,
            embedding_size=hidden_size,
            ffn_hidden=hidden_size * 4,
            n_blocks=2,
            n_heads=4,
            dataloader=self.data,
        )
        self.non_target = [
            NonTargetPlayer(input_size=self.input_size + 2, hidden_size=hidden_size)
            for _ in range(self.num_players - 1)
        ]

        # layer "all to all" to make every player aware of the other players
        self.concatenation_layer = nn.Linear(
            self.target.output_size
            + self.non_target[0].output_size * (self.num_players - 1),
            self.output_size * self.num_players,
        )

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
