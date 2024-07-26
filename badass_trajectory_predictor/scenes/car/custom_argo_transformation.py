import numpy as np
import torch

from badass_trajectory_predictor.utils import BaseTransformation


class CustomArgoTransformation(BaseTransformation):

    def __init__(self, cut, track_id, focal_track_id, cut_pos=1000):
        super().__init__()
        self.cut = cut
        self.cut_pos = cut_pos
        self.track_id = track_id
        self.focal_track_id = focal_track_id

    def forward(self, x, y, start_pos):
        if x is None:
            return None, None, None

        # Put focal_track_id to position 0
        for i in range(x.shape[0]):
            if (
                x[i, self.track_id, 0] == x[i, self.focal_track_id, 0]
                and x[i, self.focal_track_id, 0] != 0
            ):
                # Swap with i with 0
                x[[0, i]] = x[[i, 0]]
                y[[0, i]] = y[[i, 0]]
                start_pos[[0, i]] = start_pos[[i, 0]]
                break

        if torch.any(x[0, 0] == -1) or torch.any(y[0, 0] == -1):
            return None, None, None

        # [POS_X, POS_Y, TRACK_ID, OBJECT_TYPE]
        start_pos = x[:, self.cut : self.cut_pos, 0]
        return x[:, : self.cut, :], y[:, : self.cut, :], start_pos
