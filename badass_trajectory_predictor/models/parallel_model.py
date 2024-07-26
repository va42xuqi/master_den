import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from ..utils import (
    CustomDataloader,
    velocity_vector_to_position_vector,
    position_to_distance,
)


class NonTargetPlayer(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super(NonTargetPlayer, self).__init__()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.output_size = 64
        self.lstm = nn.GRU(input_size, hidden_size, 1, batch_first=True).to(device)
        # self.linear = nn.Linear(hidden_size, self.output_size).to(device)

        self.h = None

    def forward(self, input_data, reset_state=True):
        if reset_state:
            self.h = torch.zeros(1, input_data.size(0), self.lstm.hidden_size).to(
                input_data.device
            )
        lstm_output, self.h = self.lstm(input_data, self.h)
        # linear_output = self.linear(lstm_output[:, -1])
        return lstm_output[:, -1]


class ParallelModel(pl.LightningModule):
    def __init__(self, hidden_size, data: CustomDataloader, dropout=0.1):
        super(ParallelModel, self).__init__()
        self.hidden_size = hidden_size
        self.data = data
        self.input_size = 2
        self.output_size = 2
        self.num_players = data.dataset.get_input_shape()[0]
        self.pred_len = data.dataset.get_output_shape()[2]
        self.hist_len = data.dataset.get_input_shape()[2]
        self.dropout = dropout
        # self.target_function = data.target_function()
        # Create lstm layers for each player

    def forward(self, input_data, start_pos, reset_state) -> torch.Tensor:
        raise NotImplementedError

    def target_function(
        self, input_data: torch.tensor, start_pos
    ) -> list[torch.tensor]:
        # get position of player i
        court_size = (
            torch.tensor([28.65, 15.24]).unsqueeze(0).unsqueeze(2).type_as(input_data)
        )
        left_goal = (
            torch.tensor([1.22, 7.62]).unsqueeze(0).unsqueeze(2).type_as(input_data)
        )
        right_goal = (
            torch.tensor([13.11, 7.62]).unsqueeze(0).unsqueeze(2).type_as(input_data)
        )

        x_start_pos = start_pos[..., 0]
        y_start_pos = start_pos[..., 1]
        pos = self.get_pos(
            input_data.swapaxes(2, 3), x_start_pos, y_start_pos, len=input_data.shape[2]
        ).type_as(input_data)

        extra_features = []
        for i in range(self.num_players):
            is_target = i == 0

            # if target player calculate distance to both goals. If not, calculate distance to the target player
            if is_target:
                tar = pos[:, -1:]
                dist_left = tar - left_goal
                dist_right = tar - right_goal
                # standardize the distance
                dist_left = dist_left / court_size - 0.5
                dist_right = dist_right / court_size - 0.5

                extra_features.append(torch.cat((dist_left, dist_right), dim=2))
            else:
                neighbor = pos[:, :1]
                dist = pos[:, i : i + 1] - neighbor  #
                dist = dist / court_size - 0.5
                extra_features.append(dist)
        return extra_features

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, start_pos = batch

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x = torch.swapaxes(x, 2, 3)
        y = torch.swapaxes(y, 2, 3)

        x = x[..., :2]
        y = y[..., :2]

        y_hat = self(x, start_pos, reset_state=True)

        y_hat = y_hat.reshape(-1, self.num_players, 2)
        y = y[:,:,0]
        loss = nn.MSELoss()(y_hat, y)
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
        return log_dict

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, z = batch

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x = torch.swapaxes(x, 2, 3)
        y = torch.swapaxes(y, 2, 3)

        last_start_pos = x[:, :, -1, 2:]

        x = x[..., :2]
        y = y[..., :2]

        y_hat = self.test_method(x, y, z, last_start_pos)

        output_pos = self.get_pos(y[:, 0].swapaxes(1, 2))
        predict_pos = self.get_pos(y_hat.swapaxes(1, 2))

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y[:, 0].swapaxes(1, 2).cpu(),
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list

    def test_method(self, x, y, z, last_start_pos) -> torch.Tensor:
        inp = x * 1.0
        y_hat = y[:, 0] * 0.0
        for i in range(self.pred_len):
            output = self.forward(inp, start_pos=z, reset_state=i == 0)
            y_hat[:, i] = output[:, :2]
            inp = output.reshape(-1, self.num_players, 1, 2)
            z = output.reshape(-1, self.num_players, 2) * 0.04 + last_start_pos
            last_start_pos = z
        return y_hat

    def get_pos(self, inp, x_start_pos=0, y_start_pos=0, len=None):
        if len is None:
            len = self.pred_len
        return velocity_vector_to_position_vector(
            inp, [0.04] * len, x_start_pos, y_start_pos
        )

    def calculate_metrics(
        self,
        predict_pos: torch.Tensor,
        output_pos: torch.Tensor,
        vel=None,
        threshold=1e-4,
    ):
        loss_list = position_to_distance(
            np.array(predict_pos.cpu()), np.array(output_pos.cpu()), axis=1
        )

        FDE_loss = loss_list[:, -1]
        ADE_loss = loss_list.mean(axis=1)

        change_of_velocity = vel - torch.roll(vel, 1, dims=2)
        change_of_velocity[:, :, 0] = 0
        distance_between_velocity = torch.sqrt(torch.sum(change_of_velocity**2, dim=1))
        mask = distance_between_velocity > threshold

        loss_step = loss_list * np.array(mask)
        NL_ADE_loss = torch.where(
            mask.sum(dim=1) == 0,
            torch.zeros(1),
            torch.from_numpy(loss_step).sum(dim=1) / mask.sum(dim=1),
        )

        FDE_loss = np.mean(FDE_loss)
        ADE_loss = np.mean(ADE_loss)

        return FDE_loss, ADE_loss, NL_ADE_loss, loss_list

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
