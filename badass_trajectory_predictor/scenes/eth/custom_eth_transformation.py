import torch

from badass_trajectory_predictor.utils import BaseTransformation


class ETHTransform(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, start_pos):
        if x is None:
            return None, None, None

        if torch.any(x[0, 3] == -1) or torch.any(y[0, 3] == -1):
            return None, None, None

        if x[0, 6, 0] == 0:
            return None, None, None

        # Make the Frame in the startpos the first frame
        start_pos = torch.cat(
            (start_pos, x[0, 6, 0].unsqueeze(0).repeat(start_pos.shape[0], 1)), 1
        )
        return x[:, :6, :], y[:, :6, :], start_pos  # remove the label
