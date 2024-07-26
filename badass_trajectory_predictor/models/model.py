import lightning.pytorch as pl
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim

from ..utils import (
    CustomDataloader,
    velocity_vector_to_position_vector,
    position_to_distance,
)

from badass_trajectory_predictor.scenes.nba.custom_nba_transformation import (
    sliding_transformation,
)


class Model(pl.LightningModule):
    def __init__(
        self,
        dataloader: CustomDataloader,
        dropout=0.2,
        hidden_size=64,
        learning_rate=1e-3,
        history_len=50,
        pred_len=25,
    ):
        super(Model, self).__init__()
        self.input_shape = dataloader.dataset.get_input_shape()
        self.input_size = self.input_shape[0] * 2
        self.output_shape = dataloader.dataset.get_output_shape()
        self.output_size = self.input_shape[0] * 2
        self.hist_len = history_len
        self.pred_len = pred_len
        self.object_count = self.input_shape[0]
        self.num_features_in = self.input_shape[1]
        self.num_features_out = self.output_shape[1]
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate

    def forward(self, x, reset_state=True, pred_len=None) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        known_features, _, statics = batch

        if len(known_features.shape) == 3:
            known_features = known_features.unsqueeze(0)

        input_features = known_features[..., :2, : self.hist_len - 1]
        future_features = known_features[
            ..., :2, self.hist_len - 1 : self.hist_len + self.pred_len - 1
        ]
        output_features = known_features[
            ..., :2, self.hist_len : self.hist_len + self.pred_len
        ]

        x = input_features
        y = future_features

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        x = torch.cat((x, y), dim=1)
        y_hat = self.forward(x)[:, -self.pred_len :, :]
        y = output_features.permute(0, 3, 1, 2)
        y_hat = y_hat.reshape(y.shape)

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

    def validation_step(self, batch, batch_idx):
        FDE, ADE, NL_ADE, y_hat, _ = self.test_step(batch, batch_idx)

        NL_ADE = NL_ADE[NL_ADE > 1e-4].mean()

        # Use log_dict for multiple metrics
        log_dict = {"val/FDE": FDE, "val/ADE": ADE, "val/NL_ADE": NL_ADE}
        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return FDE, ADE, NL_ADE, y_hat

    def test_step(self, batch, batch_idx, pred_len=None):
        known_features, _, statics = batch
        all_y_hat = []
        all_output_features = []
        pred_len = pred_len if pred_len else self.pred_len

        input_features_list, future_features_list, _ = sliding_transformation(
            known_features, self.hist_len, pred_len
        )

        for i in range(len(input_features_list)):
            input_features = input_features_list[i][..., :2]
            output_features = future_features_list[i][..., :2]

            x = input_features
            y_hat = output_features.clone()

            inp = x.clone()

            for j in range(pred_len):
                output = self.forward(
                    inp, reset_state=j == 0, pred_len=pred_len
                ).reshape(inp.shape)[:, -1]
                y_hat[:, j] = output
                inp = torch.cat((inp, output.unsqueeze(1)), dim=1)

            all_y_hat.append(y_hat)
            all_output_features.append(output_features)

        # Concatenate all sequences along the batch dimension
        all_y_hat = torch.cat(all_y_hat, dim=0)
        all_output_features = torch.cat(all_output_features, dim=0)

        # Reshape y and y_hat
        y = all_output_features
        y = y[:, :, 0].permute(0, 2, 1)
        y_hat = all_y_hat[:, :, 0].permute(0, 2, 1)

        output_pos, predict_pos = self.get_pos(y, y_hat, pred_len)

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
            threshold=1,
        )

        MSE = nn.MSELoss()(y, y_hat)
        MAE = nn.L1Loss()(y, y_hat)

        return FDE, ADE, NL_ADE, y_hat, loss_list

    def get_pos(self, y, y_hat, pred_len=25):
        return (
            velocity_vector_to_position_vector(y.cpu(), [0.04] * pred_len, 0, 0),
            velocity_vector_to_position_vector(y_hat.cpu(), [0.04] * pred_len, 0, 0),
        )

    def calculate_metrics(self, predict_pos, output_pos, velocity=None, threshold=1e-4):
        # Calculate Euclidean distance between predicted and output positions
        loss_list = position_to_distance(
            np.array(predict_pos), np.array(output_pos), axis=1
        )

        # Final Displacement Error (FDE)
        FDE_loss = np.mean(loss_list[:, -1])

        # Average Displacement Error (ADE)
        ADE_loss = np.mean(loss_list.mean(axis=1))

        # Calculate change in velocity
        if velocity is not None:
            change_of_velocity = velocity - torch.roll(velocity, 1, dims=2)
            change_of_velocity[:, :, 0] = 0
            distance_between_velocity = torch.sqrt(
                torch.sum(change_of_velocity**2, dim=1)
            )
            mask = distance_between_velocity > threshold

            # Non-Linear Average Displacement Error (NL_ADE)
            loss_step = loss_list * np.array(mask)
            NL_ADE_loss = torch.where(
                mask.sum(dim=1) == 0,
                torch.zeros(1),
                torch.from_numpy(loss_step).sum(dim=1) / mask.sum(dim=1),
            )
        else:
            NL_ADE_loss = torch.zeros(1)

        return FDE_loss, ADE_loss, NL_ADE_loss, loss_list

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
