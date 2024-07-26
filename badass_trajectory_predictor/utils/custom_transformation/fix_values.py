from ..dataset_base import BaseTransformation
import numpy as np


class FixValues(BaseTransformation):

    def __init__(self, limit, replace, mode, channels):
        super().__init__()
        self.limit = limit
        self.replace = replace
        self.mode = mode
        self.channels = channels

    def forward(self, x, y, start_pos):
        if x is None:
            return None, None, None

        if self.mode == "greaterThan":
            for i in self.channels:
                x[:, i, :][x[:, i, :] >= self.limit] = self.replace
                y[:, i, :][y[:, i, :] >= self.limit] = self.replace
        elif self.mode == "lessThan":
            for i in self.channels:
                x[:, i, :][x[:, i, :] <= self.limit] = self.replace
                y[:, i, :][y[:, i, :] <= self.limit] = self.replace
        elif self.mode == "equal":
            for i in self.channels:
                x[:, i, :][x[:, i, :] == self.limit] = self.replace
                y[:, i, :][y[:, i, :] == self.limit] = self.replace
        elif self.mode == "notEqual":
            for i in self.channels:
                x[:, i, :][x[:, i, :] != self.limit] = self.replace
                y[:, i, :][y[:, i, :] != self.limit] = self.replace
        elif self.mode == "less":
            for i in self.channels:
                x[:, i, :][x[:, i, :] < self.limit] = self.replace
                y[:, i, :][y[:, i, :] < self.limit] = self.replace
        elif self.mode == "greater":
            for i in self.channels:
                x[:, i, :][x[:, i, :] > self.limit] = self.replace
                y[:, i, :][y[:, i, :] > self.limit] = self.replace
        else:
            raise Exception(f"Unknown mode: {self.mode}")
        return x, y, start_pos
