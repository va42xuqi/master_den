import torch
import torch.nn as nn
import random

from badass_trajectory_predictor.models.model import Model
from badass_trajectory_predictor.scenes.nba.custom_nba_transformation import (
    sliding_transformation,
)


# Linear model with 2 hidden layers and activations functions and dropout
class Linear(Model):
    def __init__(self, pred_len=None, hist_len=None, **kwargs):
        super().__init__(**kwargs)
        self.hist_len = hist_len if hist_len else self.hist_len
        self.pred_len = pred_len if pred_len else self.pred_len
        self.linear_layer = nn.Linear(
            self.input_size * 2 * (self.hist_len - 1), 2 * self.pred_len
        )
        self.num_players = 10

    def forward(self, x, **kwargs):
        # combine last two dimensions to get a 2D tensor
        x = x.reshape(x.size(0), -1)
        x = self.linear_layer(x)
        # revert back to 3D tensor
        x = x.reshape(x.size(0), self.pred_len, 2)
        return x

    def training_step(self, batch, batch_idx):
        known_features, _, statics = batch
        all_y_hat = []
        all_output_features = []

        input_features_list, future_features_list, output_features_list = (
            sliding_transformation(known_features, self.hist_len, self.pred_len)
        )

        num_sequences = len(input_features_list)
        range_sequences = list(range(num_sequences))
        random.shuffle(range_sequences)

        for i in range_sequences[:5]:
            input_features = input_features_list[i]
            output_features = future_features_list[i][..., 0, :2]

            y_hat = self(input_features)
            all_y_hat.append(y_hat)
            all_output_features.append(output_features)

        y_hat = torch.cat(all_y_hat, dim=0)
        output_features = torch.cat(all_output_features, dim=0)

        y = output_features

        loss = nn.MSELoss()(y, y_hat)
        # more losses
        mae = nn.L1Loss()(y, y_hat)
        smae = nn.SmoothL1Loss()(y, y_hat)
        log_cosh = torch.log(torch.cosh(y_hat - y)).mean()

        log_dict = {
            "train/mse": loss,
            "train/mae": mae,
            "train/smae": smae,
            "train/log_cosh": log_cosh,
        }

        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx, pred_len=5, **kwargs):
        known_features, _, statics = batch
        all_y_hat = []
        all_output_features = []

        input_features_list, future_features_list, output_features_list = (
            sliding_transformation(known_features, self.hist_len, self.pred_len)
        )

        for i in range(len(input_features_list)):
            input_features = input_features_list[i]
            output_features = future_features_list[i][..., 0, :2]

            y_hat = self(input_features)
            all_y_hat.append(y_hat)
            all_output_features.append(output_features)

        y_hat = torch.cat(all_y_hat, dim=0).permute(0, 2, 1)
        y = torch.cat(all_output_features, dim=0).permute(0, 2, 1)

        # check the first player

        output_pos, predict_pos = self.get_pos(y, y_hat, pred_len=pred_len)

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list
