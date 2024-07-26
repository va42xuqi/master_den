import torch
import torch.nn as nn

from badass_trajectory_predictor.models.model import Model


class SingleLayerCNN(Model):
    def __init__(self, **kwargs):
        super(SingleLayerCNN, self).__init__(**kwargs)
        # Define the convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=self.input_size, out_channels=64, kernel_size=3, padding=1
        )
        self.readout = nn.Linear(64, self.input_size)
        self.num_players = 10

    def forward(self, x, **kwargs):
        # combine the last two dimensions to get a 2D tensor
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = self.readout(x)
        x = x.permute(0, 2, 1)
        return x[:, :, -self.pred_len :]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        x = x[:, :, :2]
        y = y[:, :, :2]

        x = x.reshape((x.size(0), -1, x.size(-1)))
        y = y.reshape((y.size(0), -1, y.size(-1)))

        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
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

        x = x.reshape((x.size(0), -1, x.size(-1)))

        y_hat = self.forward(x)
        y_hat = y_hat.reshape((-1, self.num_players, 2, self.pred_len))

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
