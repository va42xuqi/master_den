import torch
from torch import nn

from badass_trajectory_predictor.models.model import Model
from badass_trajectory_predictor.models.attention_based.retentive_utils.retnet import RetNet as RetNetLayer


class RetNet(Model):
    def __init__(
        self,
        generator,
        pos_encoding=None,
        layers=24,
        hidden_dim=2048,
        ffn_size=4096,
        heads=16,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        layers = layers
        hidden_dim = hidden_dim
        ffn_size = ffn_size
        heads = heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm = nn.LayerNorm(hidden_dim)
        self.pos_encoding = pos_encoding
        self.retnet = RetNetLayer(
            layers, hidden_dim, ffn_size, heads, double_v_dim=True
        ).to(device)
        self.generator = generator

    def forward(self, x, **kwargs):
        x = self.pos_encoding(x)
        x = self.retnet(x)
        x = self.norm(x)
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        x = x[:, :, :2]
        y = y[:, :, :2]

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        # teacher forcing
        input = torch.cat((x, y[:, :-1]), dim=1)

        y_hat = self(input)

        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx % 100 != 0:
            return None
        FDE, ADE, NL_ADE, y_hat, _ = self.test_step(batch, batch_idx, pred_len=5)

        NL_ADE = NL_ADE[NL_ADE > 1e-4].mean()

        # Use log_dict for multiple metrics
        log_dict = {"FDE": FDE, "ADE": ADE, "NL_ADE": NL_ADE}
        self.log_dict(log_dict, sync_dist=True)

    def test_step(self, batch, batch_idx, pred_len=None):
        x, y, _ = batch

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x = x[:, :, :2]
        y = y[:, :, :2]

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        y_hat = y.clone()
        inp = x.clone()

        for i in range(self.pred_len if pred_len is None else pred_len):
            output = self.forward(inp, reset_state=i == 0)[:, 0]
            y_hat[:, i] = output
            inp = torch.cat((inp, output.unsqueeze(1)), dim=1)

        output_pos, predict_pos = self.get_pos(y.swapaxes(2, 3), y_hat.swapaxes(2, 3))

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.swapaxes(2, 3).cpu(),
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list
