from ..dataset_base import BaseTransformation
import torch


class SortTrajectories(BaseTransformation):

    def __init__(self, comparator=None, start_dim=2):
        super().__init__()
        if comparator is None:
            comparator = arg_sort
        self.comparator = comparator
        self.start_dim = start_dim

    def forward(self, x, y, start_pos):
        if x is None:
            return None, None, None

        distances = torch.sqrt(
            torch.sum(
                (
                    x[:, self.start_dim : self.start_dim + 2, -1]
                    - x[:1, self.start_dim : self.start_dim + 2, -1]
                )
                ** 2,
                dim=1,
            )
        )

        sorted_indices = self.comparator(distances)
        x = x[sorted_indices]
        y = y[sorted_indices]
        start_pos = start_pos[sorted_indices]
        return x, y, start_pos


def arg_sort(distances):
    return torch.argsort(distances)
