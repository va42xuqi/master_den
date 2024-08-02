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
from codecarbon import EmissionsTracker


def get_angular_error(y, y_hat):
    """
    Calculate the angular error between the predicted and ground truth velocity vectors.

    Parameters:
    y (torch.Tensor): Ground truth velocity vectors of shape (batch_size, seq_length, n, 2)
    y_hat (torch.Tensor): Predicted velocity vectors of shape (batch_size, seq_length, n, 2)

    Returns:
    torch.Tensor: Angular error between the predicted and ground truth velocity vectors.
    """
    # Calculate the dot product between the predicted and ground truth velocity vectors
    dot_product = torch.sum(y * y_hat, dim=-2)

    # Calculate the magnitudes of the predicted and ground truth velocity vectors
    magnitude_y = torch.norm(y, dim=-2)
    magnitude_y_hat = torch.norm(y_hat, dim=-2)

    # Calculate the cosine similarity between the predicted and ground truth velocity vectors
    cosine_similarity = dot_product / (magnitude_y * magnitude_y_hat + 1e-10)

    # Clamp the cosine similarity to the valid range [-1, 1]
    cosine_similarity = torch.clamp(cosine_similarity, min=-1, max=1)

    # Calculate the angular error between the predicted and ground truth velocity vectors
    angular_error = torch.acos(cosine_similarity)

    return angular_error


def random_rotate_blocks(tensor, dim=2):
    """
    Randomly rotate blocks along the specified dimension.

    Parameters:
    tensor (torch.Tensor): Input tensor of shape (batch_size, seq_length, n, ...)
    dim (int): Dimension to rotate along. Default is 2.

    Returns:
    torch.Tensor: Tensor with randomly rotated blocks.
    """
    # Split the tensor along the specified dimension
    first_half, second_half = torch.chunk(tensor, 2, dim=dim)

    # Randomly permute elements within each half along the specified dimension
    first_half_permuted = first_half[:, torch.randperm(first_half.size(dim))]
    second_half_permuted = second_half[:, torch.randperm(second_half.size(dim))]

    # Concatenate the randomly permuted halves to form the final tensor
    rotated_tensor = torch.cat((first_half_permuted, second_half_permuted), dim=dim)

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


def importance(x, a=3, b=3):
    pass


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
        self.train_tracker = EmissionsTracker(
            log_level="error", measure_power_secs=10, tracking_mode="process"
        )
        self.val_tracker = EmissionsTracker(
            log_level="error", measure_power_secs=10, tracking_mode="process"
        )
        self.test_tracker = EmissionsTracker(
            log_level="error", measure_power_secs=10, tracking_mode="process"
        )

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

    def on_train_start(self):
        self.train_tracker.start_task("train")

    def on_train_end(self):
        emissions = self.train_tracker.stop_task("train")

        # Extract relevant metrics from EmissionsData
        total_energy = emissions.energy_consumed  # Total energy consumed in kWh
        gpu_energy = emissions.gpu_energy  # GPU energy consumption in kWh
        cpu_energy = emissions.cpu_energy  # CPU energy consumption in kWh
        ram_energy = emissions.ram_energy  # RAM energy consumption in kWh

        # Log scalar values
        if self.logger:
            self.logger.experiment.log(
                {
                    "Total energy consumption (kWh)": total_energy,
                    "GPU energy consumption (kWh)": gpu_energy,
                    "CPU energy consumption (kWh)": cpu_energy,
                    "RAM energy consumption (kWh)": ram_energy,
                }
            )

    def on_validation_epoch_start(self):
        self.val_tracker.start_task("validation")

    def on_validation_epoch_end(self):
        emissions = self.val_tracker.stop_task("validation")

        # Extract relevant metrics from the EmissionsData object
        total_energy = emissions.energy_consumed  # Total energy consumed in kWh
        gpu_energy = emissions.gpu_energy  # GPU energy consumption in kWh
        cpu_energy = emissions.cpu_energy  # CPU energy consumption in kWh
        ram_energy = emissions.ram_energy  # RAM energy consumption in kWh


        # Log scalar values
        self.log_dict(
            {
                "Total energy consumption (kWh)": total_energy,
                "GPU energy consumption (kWh)": gpu_energy,
                "CPU energy consumption (kWh)": cpu_energy,
                "RAM energy consumption (kWh)": ram_energy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_start(self):
        self.test_tracker.start_task("test")

    def on_test_epoch_end(self):
        emissions = self.test_tracker.stop_task("test")
        
        # Extract relevant metrics from the EmissionsData object
        total_energy = emissions.energy_consumed  # Total energy consumed in kWh
        gpu_energy = emissions.gpu_energy  # GPU energy consumption in kWh
        cpu_energy = emissions.cpu_energy  # CPU energy consumption in kWh
        ram_energy = emissions.ram_energy  # RAM energy consumption in kWh

        # Log scalar values
        self.log_dict(
            {
                "Total energy consumption (kWh)": total_energy,
                "GPU energy consumption (kWh)": gpu_energy,
                "CPU energy consumption (kWh)": cpu_energy,
                "RAM energy consumption (kWh)": ram_energy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def step(self, batch, num_batches=10, shuffle=True):
        known_features, _, statics = batch

        num_obj = known_features.size(1)
        if self.has_ball:
            num_obj -= 1
        if shuffle:
            known_features[:, :num_obj] = random_rotate_blocks(
                known_features[:, :num_obj], dim=1
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

    def test_step(self, batch, batch_idx, **kwargs):
        loss, output, x, y = self.step(batch, shuffle=False, num_batches=-1)

        z = y[:, 0]

        angular_error = get_angular_error(z, output).detach().cpu()

        FRE = angular_error[:, -1] * 180 / np.pi  # Final Radian Error
        ARE = angular_error.detach() * 180 / np.pi  # Average Radian Error

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
        basket * sign.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
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
