import numpy as np

from ..dataset_base import BaseTransformation


class LastTransformation(BaseTransformation):

    def __init__(self, delete_features=None):
        super().__init__()
        self.delete_features = delete_features

    def forward(self, x, y, startpos):
        if x is None:
            return None, None, None

        # if delete_features is not None delete the features
        if self.delete_features is not None:
            x = np.delete(x, self.delete_features, axis=1)
            y = np.delete(y, self.delete_features, axis=1)

        return x, y, startpos
