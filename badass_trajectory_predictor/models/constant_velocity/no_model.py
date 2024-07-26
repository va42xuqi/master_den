import torch
import torch.nn as nn

from badass_trajectory_predictor.models.model import Model


class NoModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        # it's the last value from the input (prediction times)
        last_timestep = x[:, -1, :]
        repeated_last_timestep = last_timestep.unsqueeze(1).repeat(1, self.pred_len, 1)
        return repeated_last_timestep

    def training_step(self, batch, batch_idx, **kwargs):
        x, y, _ = batch

        x = x[:, 0]
        y = y[:, 0]
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        # weight input such as velocity is more important than position (99 % velocity, 1 % position)
        weighted_input = x * torch.tensor([1, 1, 0.0, 0.0]).to(x.device)
        y_hat = self(weighted_input)

        y = y[:, :, :2]
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x = x[:, 0]
        x = x.permute(0, 2, 1)

        y = y[:, 0, :2]
        y = y.permute(0, 2, 1)

        inp = x.clone()
        weighted_input = inp * torch.tensor([1, 1, 0.0, 0.0]).to(inp.device)
        y_hat = self.forward(weighted_input)[..., -self.pred_len :]

        y_hat = y_hat.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        output_pos, predict_pos = self.get_pos(y, y_hat)

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list
