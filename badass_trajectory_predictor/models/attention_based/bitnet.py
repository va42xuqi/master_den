import torch
from torch import nn
from torch.optim import Adam

from badass_trajectory_predictor.models.model import Model
from badass_trajectory_predictor.models.attention_based.bitnet_utils.bitnet import (
    GPTLike as BitNetLayer,
)
from badass_trajectory_predictor.models.attention_based.transformer_utils.helper import (
    subsequent_mask,
)


class BitNet(Model):
    def __init__(
        self,
        n_blks=6,
        embed_dim=128,
        ffn_hiddens=64,
        n_heads=8,
        pos_encoding=nn.Identity(),
        generator=nn.Identity(),
        **kwargs
    ) -> None:
        self.embed_size = embed_dim
        super().__init__(**kwargs)
        self.tf = BitNetLayer(
            n_blks=n_blks,
            input_dim=embed_dim,
            ffn_hiddens=ffn_hiddens,
            n_heads=n_heads,
            dropout=self.dropout,
            pos_encoding=pos_encoding,
            generator=generator,
        )

    def forward(self, x, tgt_mask=None, **kwargs):
        out = self.tf(x, tgt_mask=tgt_mask)
        return out

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        x = x[:, :, :2]
        y = y[:, :, :2]

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        # teacher forcing
        input = torch.cat((x, y[:, :-1]), dim=1)

        tgt_mask = subsequent_mask(input.size(1) * self.object_count).to(input.device)
        y_hat = self(input, tgt_mask=tgt_mask)

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
            # weight input such as velocity is more important than position (90 % velocity, 10 % position)
            output = self.forward(inp, reset_state=i == 0)[:, 0]
            y_hat[:, i] = output
            inp = torch.cat((inp, output.unsqueeze(1)), dim=1)

        y = y.swapaxes(2, 3)
        y_hat = y_hat.swapaxes(2, 3)

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

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
