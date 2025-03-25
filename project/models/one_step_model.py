import lightning.pytorch as pl
import torch
import random
import numpy as np
from project.scenes.nba.custom_nba_transformation import (
    sliding_transformation,
)
from ..utils import (
    velocity_vector_to_position_vector,
    position_to_distance,
)
from torch import nn
import torch
import logging
logging.basicConfig(level=logging.INFO)


def get_angular_error(y, y_hat):
    dot_product = torch.sum(y_hat * y, dim=0)
    cross_product = y_hat[0] * y[1] - y_hat[1] * y[0]

    magnitude_x = torch.norm(y_hat, dim=0)
    magnitude_y = torch.norm(y, dim=0)

    cos_angle = dot_product / (magnitude_x * magnitude_y).clamp(min=1e-8)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle = torch.where(
        (magnitude_x < 1e-8) | (magnitude_y < 1e-8),
        torch.zeros_like(cos_angle),
        torch.acos(cos_angle)
    )
    angle = torch.where(cross_product < 0, -angle, angle)

    average_angular_error = torch.mean(angle, dim=-1)
    return average_angular_error, angle

def random_rotate_blocks(tensor, dim=2, test_on_other_team=False, shuffle=True):
    first_half, second_half = torch.chunk(tensor, 2, dim=dim)

    if shuffle:
        indices_first = torch.randperm(first_half.size(dim), device=tensor.device)
        indices_second = torch.randperm(second_half.size(dim), device=tensor.device)

        first_half = first_half.index_select(dim, indices_first)
        second_half = second_half.index_select(dim, indices_second)

    if test_on_other_team:
        rotated_tensor = torch.cat((second_half, first_half), dim=dim)
    else:
        rotated_tensor = torch.cat((first_half, second_half), dim=dim)

    return rotated_tensor


def relative_position(x):
    target = x[:, :, 0, 2:].clone()
    x = x[:, :, 1:, 2:].clone()
    x = x - target.unsqueeze(2)
    return x


def pos_to_basket(x, basket_positions):
    pos = x.clone()
    dist = basket_positions - pos
    return dist


def has_ball_flags(ball, players):
    ball = ball[..., 2:].clone()
    team = players[:, :, :5, 2:].clone()
    opponent = players[:, :, 5:, 2:].clone()

    distance_team = torch.norm(team - ball, dim=-1)
    distance_opponent = torch.norm(opponent - ball, dim=-1)

    team_flag = torch.any(distance_team < 1, dim=-1, keepdim=True).long()
    opponent_flag = torch.any(distance_opponent < 1, dim=-1, keepdim=True).long()

    return torch.cat([team_flag, opponent_flag], dim=-1)


class OneStepModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        history_len,
        prediction_len,
        num_players,
        config,
        dropout=0.2,
        has_ball=False,
        has_goals=False,
        pretrain=False,
        fine_tune=False,
    ):
        super(OneStepModel, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.output_size = 2
        self.non_targets = num_players - 1
        self.has_ball = has_ball
        self.num_players = num_players if not (pretrain or fine_tune) else 1
        self.convert_to_centered = config.convert_to_centered
        self.get_goal_position = config.get_goal_position
        self.in_features = 4 * (self.num_players + has_ball + 2 * has_goals)
        self.has_goals = has_goals

        basket = torch.tensor(self.get_goal_position())
        basket_vel = torch.zeros_like(basket)
        self.basket_features = torch.cat([basket_vel, basket], dim=-1).repeat(
            self.history_len, 1, 1
        )
        self.pretrain = (pretrain,)
        self.fine_tune = fine_tune

        self.best_ADE = 1000
        self.best_NL_ADE = 1000
        self.best_FDE = 1000

    def forward(self, src, statics):
        pass

    def step(self, batch, num_batches=10, shuffle=True, test_on_other_team=False):
        known_features, _, statics = batch

        num_obj = known_features.size(1)
        if self.has_ball:
            num_obj -= 1
        if shuffle or test_on_other_team:
            known_features[:, :num_obj] = random_rotate_blocks(
                known_features[:, :num_obj], dim=1, test_on_other_team=test_on_other_team, shuffle=shuffle
            )

        statics = statics.unsqueeze(0) if len(statics.shape) == 2 else statics

        input_features_list, future_features_list, output_features_list = (
            sliding_transformation(
                known_features, self.history_len, self.prediction_len
            )
        )

        all_y_hat = []
        all_output_features = []
        all_x = []

        num_sequences = len(input_features_list)
        range_sequences = list(range(num_sequences))

        if shuffle:
            random.shuffle(range_sequences)

        if num_batches > 0:
            range_sequences = range_sequences[:num_batches]

        for i in range_sequences:
            input_features = input_features_list[i]
            output_features = future_features_list[i][..., :2]

            inp = input_features.clone()

            output = self(inp, statics)
            y = output_features

            all_y_hat.append(output.permute(0, 2, 1))
            all_output_features.append(y.permute(0, 2, 3, 1))
            all_x.append(input_features.permute(0, 2, 3, 1))

        all_y_hat = torch.cat(all_y_hat, dim=0)
        all_output_features = torch.cat(all_output_features, dim=0)
        all_x = torch.cat(all_x, dim=0)

        error = nn.MSELoss()(all_y_hat, all_output_features[:, 0])

        # angular_error = get_angular_error(all_output_features[:, 0], all_y_hat) # not implemented

        loss = error

        return loss, all_y_hat, all_x, all_output_features

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, x, y = self.step(batch)

        y = y[:, 0]
        output_pos, predict_pos = self.get_pos(y, output, pred_len=self.prediction_len)

        FDE, ADE, NL_ADE, MSE, MAE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
            threshold=0.5,
        )

        NL_ADE = NL_ADE[NL_ADE > 0.5]

        self.log("val/FDE", FDE.mean(), on_step=True, on_epoch=True)
        self.log("val/ADE", ADE.mean(), on_step=True, on_epoch=True)
        self.log("val/NL_ADE", NL_ADE.mean(), on_step=True, on_epoch=True)

        if FDE.mean() < self.best_FDE:
            self.best_FDE = FDE.mean()
            self.log("val/best_FDE", FDE.mean(), on_step=True, on_epoch=True)
        
        if ADE.mean() < self.best_ADE:
            self.best_ADE = ADE.mean()
            self.log("val/best_ADE", ADE.mean(), on_step=True, on_epoch=True)

        if NL_ADE.mean() < self.best_NL_ADE:
            self.best_NL_ADE = NL_ADE.mean()
            self.log("val/best_NL_ADE", NL_ADE.mean(), on_step=True, on_epoch=True)

        return FDE

    def test_step(self, batch, batch_idx, test_on_other_team, **kwargs):
        loss, output, x, y = self.step(batch, shuffle=False, num_batches=-1, test_on_other_team=test_on_other_team)

        z = y[:, 0]  # Ground truth positions

        output = output.cpu()  # Predicted positions
        z = z.cpu()  # Ground truth positions

        # Calculate angular errors
        ARE, angular_error = get_angular_error(z, output)

        # Final angular error for the last prediction step
        FRE = angular_error[:, 49]  # Shape: (batch_size,)

        output_pos, predict_pos = self.get_pos(z, output, pred_len=self.prediction_len)

        FDE, ADE, NL_ADE, MSE, MAE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            z.cpu(),
            threshold=0.5,
        )

        NL_ADE = NL_ADE[NL_ADE > 0.5]

        return (
            FDE,
            ADE,
            NL_ADE,
            MSE,
            MAE,
            FRE,
            ARE,
            loss_list,
            angular_error,
            output,
            x,
            y,
        )

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optim

    def preprocess_data(self, data):
        src, statics = data
        src = torch.cat([src[:, :, : self.num_players], src[:, :, -1:]], dim=2)
        statics = torch.cat([statics[:, : self.num_players], statics[:, -1:]], dim=1)
        basket = (
            self.basket_features.to(src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1, 1, 1)
        )
        sign = statics[:, 0, 0] * 2 - 1
        basket = basket * sign.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = torch.cat([src, basket], dim=2)
        return out, statics

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
        FDE_loss = loss_list[:, -1]
        # Average Displacement Error (ADE)
        ADE_loss = loss_list.mean(axis=1)
        # Compute element-wise squared differences
        squared_diff = (predict_pos - output_pos) ** 2
        MSE = squared_diff.mean(dim=(1, 2))
        abs_diff = torch.abs(predict_pos - output_pos)
        MAE = abs_diff.mean(dim=(1, 2))

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

        return FDE_loss, ADE_loss, NL_ADE_loss, MSE, MAE, loss_list