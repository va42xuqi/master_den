from project.utils.dataset_base import BaseTransformation
import numpy as np
import torch


class CustomSoccerTransformation(BaseTransformation):

    def __init__(self, move=None):
        super().__init__()
        self.move = move

    def forward(self, x, y, start_pos):
        # Move Team_id and Player_id to pos
        if x is None:
            return None, None, None

        if self.move:
            if int(x[0, 6, 0].item()) == int(self.move, 36):
                # swap with player 0 with player 1
                x[0], x[1] = x[1], x[0]
                y[0], y[1] = y[1], y[0]
                start_pos[0], start_pos[1] = start_pos[1], start_pos[0]

        start_pos = torch.cat((start_pos, x[:, 5, 0].unsqueeze(1), x[:, 6, 0].unsqueeze(1)), 1)
        return x[:, :5, :], y[:, :5, :], start_pos[:, 2:]

    def to_string(self):
        return f"CustomSoccerTransformation(move={self.move})"
