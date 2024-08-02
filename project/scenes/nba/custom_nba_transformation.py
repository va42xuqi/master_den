import numpy as np
import torch

from project.utils import BaseTransformation


class CustomNBATransformation(BaseTransformation):
    def __init__(self, move=None):
        super().__init__()
        self.move = move

    def forward(self, x, y, start_pos):
        # Move Team_id and Player_id to pos
        if x is None:
            return None, None, None

        start_pos = torch.cat(
            (
                start_pos,
                x[:, 5, 0].unsqueeze(1),
                x[:, 6, 0].unsqueeze(1),
            ),
            1,
        )  # teamID, RoleID, PlayerID
        return x[:, :5, :], y[:, :5, :], start_pos[:, 2:]


def sliding_transformation(x, hl, pl):
    x = x.unsqueeze(0) if len(x.shape) == 3 else x
    x = x.permute(0, 3, 1, 2)

    input_features = []
    future_features = []
    output_features = []
    for i in range(x.shape[1] - hl - pl + 1):
        input_features.append(x[:, i : i + hl])
        future_features.append(x[:, i + hl : i + hl + pl])

    return input_features, future_features, output_features
