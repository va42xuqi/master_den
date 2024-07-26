import torch
import torch.nn as nn

from badass_trajectory_predictor.models.model import Model


# Linear model with 2 hidden layers and activations functions and dropout
class LinearDO(Model):
    def __init__(self, num_hidden_layers=2, weight_decay=1e-5, **kwargs):
        hidden_size = 2**12
        super().__init__(**kwargs)
        self.linear_layer = nn.Linear(self.input_size * self.hist_len, hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.linear_layer2 = nn.Linear(hidden_size, self.input_size * self.pred_len)
        self.num_players = 10

    def forward(self, x, **kwargs):
        # combine last two dimensions to get a 2D tensor
        x = x.reshape(x.size(0), -1)
        x = self.linear_layer(x)
        x = self.dropout(x)
        x = self.linear_layer2(x)
        # revert back to 3D tensor
        x = x.reshape(x.size(0), self.pred_len, self.output_size)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        x = x[:, :, :2]
        y = y[:, :, :2]

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        x = x.reshape((x.size(0), x.size(1), -1))
        y = y.reshape((y.size(0), y.size(1), -1))

        y_hat = self.forward(x)
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

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, _ = batch

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x = x[:, :, :2]
        y = y[:, :, :2]

        x = x.permute(0, 3, 1, 2)
        x = x.reshape((x.size(0), x.size(1), -1))

        y_hat = self.forward(x)
        y_hat = y_hat.swapaxes(1, 2)
        y_hat = y_hat.reshape(-1, self.num_players, 2, self.pred_len)

        # check the first player

        output_pos, predict_pos = self.get_pos(y, y_hat)

        y = y[:, 0]
        output_pos = output_pos[:, 0]
        predict_pos = predict_pos[:, 0]

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list
