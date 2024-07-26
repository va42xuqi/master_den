import torch
from torch import nn
from torch.optim import AdamW
import random

from badass_trajectory_predictor.models.model import Model
from . import TrafoLayer
from . import subsequent_mask
from .transformer_utils import NoamOpt

from badass_trajectory_predictor.scenes.nba.custom_nba_transformation import (
    sliding_transformation,
)


class Trafo(Model):
    def __init__(
        self,
        n_blocks=6,
        embed_dim=128,
        ffn_hidden=64,
        n_heads=8,
        pos_encoding=nn.Identity(),
        generator=nn.Identity(),
        pred_length=25,
        history_length=50,
        **kwargs
    ) -> None:
        super(Trafo, self).__init__(**kwargs)
        self.pred_len = pred_length
        self.hist_len = history_length
        self.embed_size = embed_dim

        self.tf = TrafoLayer(
            n_blocks=n_blocks,
            input_dim=embed_dim,
            ffn_hidden=ffn_hidden,
            n_heads=n_heads,
            dropout=self.dropout,
            pos_encoding=pos_encoding,
            generator=generator,
        )

    def forward(self, x, tgt_mask=None, **kwargs):
        out = self.tf(x.reshape(x.size(0), x.size(1), -1), tgt_mask=tgt_mask)
        return out

    def training_step(self, batch, batch_idx):
        known_features, _, statics = batch

        input_features_list, future_features_list, output_features_list = (
            sliding_transformation(known_features, self.hist_len, self.pred_len)
        )

        total_loss = 0
        total_mae = 0
        total_smae = 0
        total_log_cosh = 0
        num_sequences = len(input_features_list)
        range_sequences = list(range(num_sequences))

        random.shuffle(range_sequences)

        for i in range_sequences[:20]:
            input_features = input_features_list[i][..., :2]
            output_features = output_features_list[i][..., :2]
            future_features = future_features_list[i][..., :2]

            x = input_features
            y = future_features

            # Teacher forcing
            input = torch.cat((x, y), dim=1)
            tgt_mask = subsequent_mask(input.size(1)).to(input.device)
            y_hat = self(input, tgt_mask=tgt_mask).reshape(input.shape)[
                :, -self.pred_len :
            ]
            y = output_features

            loss = nn.MSELoss()(y, y_hat)
            mae = nn.L1Loss()(y, y_hat)
            smae = nn.SmoothL1Loss()(y, y_hat)
            log_cosh = torch.log(torch.cosh(y_hat - y)).mean()

            total_loss += loss
            total_mae += mae
            total_smae += smae
            total_log_cosh += log_cosh

            torch.cuda.empty_cache()

        # Average the losses
        avg_loss = total_loss / num_sequences
        avg_mae = total_mae / num_sequences
        avg_smae = total_smae / num_sequences
        avg_log_cosh = total_log_cosh / num_sequences

        log_dict = {
            "train/mse": avg_loss,
            "train/mae": avg_mae,
            "train/smae": avg_smae,
            "train/log_cosh": avg_log_cosh,
        }

        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        return avg_loss

    def validation_step(self, batch, batch_idx):
        if batch_idx % 100 != 0:
            return None
        FDE, ADE, NL_ADE, y_hat, _ = self.test_step(batch, batch_idx, pred_len=5)
        NL_ADE = NL_ADE[NL_ADE > 1e-4].mean()

        # Use log_dict for multiple metrics
        log_dict = {"val/FDE": FDE, "val/ADE": ADE, "val/NL_ADE": NL_ADE}
        self.log_dict(log_dict, sync_dist=True)

    def test_step(self, batch, batch_idx, pred_len=None):
        known_features, _, statics = batch
        all_y_hat = []
        all_output_features = []

        input_features_list, future_features_list, _ = sliding_transformation(
            known_features, self.hist_len, self.pred_len
        )

        for i in range(len(input_features_list)):
            input_features = input_features_list[i][..., :2]
            output_features = future_features_list[i][..., :2]

            x = input_features
            y_hat = output_features.clone()

            inp = x.clone()

            for j in range(self.pred_len if pred_len is None else pred_len):
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

        output_pos, predict_pos = self.get_pos(y, y_hat)

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
            threshold=1,
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamOpt(
            model_size=self.embed_size, factor=0.5, warmup=4000, optimizer=optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
