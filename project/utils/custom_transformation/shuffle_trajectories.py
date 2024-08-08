from project.utils.dataset_base import BaseTransformation
import torch


class ShuffleTrajectories(BaseTransformation):

    def __init__(self, sure_shuffle=-1, n_players=5):
        super().__init__()
        self.sure_shuffle = sure_shuffle
        self.n_players = n_players

    def forward(self, x, y, start_pos):
        # erste n Spieler mischen
        indices_low = torch.randperm(self.n_players)
        x[: self.n_players] = x[indices_low]
        y[: self.n_players] = y[indices_low]
        start_pos[: self.n_players] = start_pos[indices_low]

        indices_high = torch.randperm(x.shape[0] - self.n_players) + self.n_players
        x[self.n_players :] = x[indices_high]
        y[self.n_players :] = y[indices_high]
        start_pos[self.n_players :] = start_pos[indices_high]

        return x, y, start_pos

    def to_string(self):
        return f"ShuffleTrajectories(sure_shuffle={self.sure_shuffle}, n_players={self.n_players})"
