import torch
import torch.nn as nn
import numpy as np
import random

from badass_trajectory_predictor.models.model import Model
from badass_trajectory_predictor.models.attention_based import NoamOpt
from badass_trajectory_predictor.models.attention_based.transformer_utils import (
    TimeSeriesEmbedding,
)
from badass_trajectory_predictor.scenes.nba import sliding_transformation


# Shared Preprocessing LSTM module for neighbor objects
class SharedPreprocessingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SharedPreprocessingLSTM, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, h_n = self.gru(x)
        return out


class TargetTransformer(nn.Module):
    def __init__(
        self,
        target_size,
        hidden_size,
        output_size,
        n_head,
        num_decoder_layer,
        seq_length=1000,
    ):
        super().__init__()
        self.embedding = TimeSeriesEmbedding(target_size, hidden_size, seq_length)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_head, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layer
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, memory):
        x = self.embedding(x)
        x = self.transformer_decoder(x, memory)
        return self.linear(x[:, -1])


class SupTrafo(Model):
    def __init__(self, n_head=2, num_decoder_layer=2, **kwargs):
        super().__init__(**kwargs)
        self.target_size = 2
        self.input_size = 2
        self.output_size = 2
        hidden_size = self.hidden_size
        player_size = 10

        self.shared_preprocessing_transformer = SharedPreprocessingLSTM(
            self.input_size + 2, self.hidden_size
        )
        self.target_preprocessing_transformer = SharedPreprocessingLSTM(
            self.target_size + 2, self.hidden_size
        )
        self.combining_linear = nn.Linear(hidden_size * player_size, hidden_size)
        self.target_transformer = TargetTransformer(
            self.target_size,
            hidden_size,
            self.output_size,
            n_head,
            num_decoder_layer,
            seq_length=1000,
        )

    def preprocess_objects(self, data):
        num_players, batch_size, seq_len, input_dim = data.size()
        target_data = data[0]
        non_target_data = data[1:].reshape(
            batch_size * (num_players - 1), seq_len, input_dim
        )

        target_hidden_state = self.target_preprocessing_transformer(target_data)
        nt_hidden_state = self.shared_preprocessing_transformer(non_target_data)
        nt_hidden_states = nt_hidden_state.reshape(batch_size, seq_len, -1)
        combined_hidden_states = torch.cat(
            [nt_hidden_states, target_hidden_state], dim=2
        )
        transformed_state = self.combining_linear(combined_hidden_states)
        return transformed_state

    def forward(self, data, target=None, teacher_forcing=True, pred_len=None):
        batch_size = data.size(0)
        pred_len = pred_len if pred_len else self.pred_len
        outputs = torch.zeros(
            batch_size, pred_len, self.target_size, device=data[0].device
        )

        data = data.permute(2, 0, 1, 3)

        combined_hidden_state = self.preprocess_objects(data)
        input_seq = target[:, :1, :]

        for t in range(pred_len):
            output = self.target_transformer(input_seq, combined_hidden_state)
            outputs[:, t, :] = output
            if t + 1 < pred_len and teacher_forcing and target is not None:
                next_seq = target[:, t + 1, :]
            else:
                next_seq = output
            input_seq = torch.cat([input_seq, next_seq.unsqueeze(1)], dim=1)

        return outputs

    def training_step(self, batch, batch_idx):
        known_features, _, statics = batch
        all_y_hat = []
        all_output_features = []

        input_features_list, future_features_list, output_features_list = (
            sliding_transformation(known_features, self.hist_len, self.pred_len)
        )

        num_sequences = len(input_features_list)
        range_sequences = list(range(num_sequences))
        random.shuffle(range_sequences)

        for i in range_sequences[:5]:
            input_features = input_features_list[i]
            future_features = future_features_list[i][..., 0, :2]
            output_features = output_features_list[i][..., 0, :2]

            y_hat = self(input_features, future_features, teacher_forcing=True)
            y = output_features

            all_y_hat.append(y_hat)
            all_output_features.append(y)

        output = torch.cat(all_y_hat, dim=0)
        y = torch.cat(all_output_features, dim=0)

        loss = nn.MSELoss()(y, output)
        mae = nn.L1Loss()(y, output)
        smae = nn.SmoothL1Loss()(y, output)
        log_cosh = torch.log(torch.cosh(output - y)).mean()

        log_dict = {
            "train/mse": loss,
            "train/mae": mae,
            "train/smae": smae,
            "train/log_cosh": log_cosh,
        }

        self.log_dict(
            log_dict, sync_dist=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        FDE, ADE, NL_ADE, y_hat, _ = self.test_step(batch, batch_idx, pred_len=5)
        NL_ADE = NL_ADE[NL_ADE > 0.5].mean()
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

        input_features_list, future_features_list, output_features_list = (
            sliding_transformation(known_features, self.hist_len, pred_len)
        )

        for i in range(len(input_features_list)):
            input_features = input_features_list[i]
            future_features = future_features_list[i][:, :, 0, :2]
            output_features = output_features_list[i][:, :, 0, :2]

            y_hat = self(
                input_features,
                future_features,
                teacher_forcing=False,
                pred_len=pred_len,
            )
            y_hat = y_hat.permute(0, 2, 1)
            y = output_features.permute(0, 2, 1)

            all_y_hat.append(y_hat)
            all_output_features.append(y)

        y_hat = torch.cat(all_y_hat, dim=0)
        y = torch.cat(all_output_features, dim=0)

        output_pos, predict_pos = self.get_pos(y, y_hat, pred_len)
        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos, output_pos, y.cpu(), threshold=0.5
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list

    def configure_optimizers(self):
        cuda = torch.cuda.is_available()
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=0.001,
            foreach=cuda,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01,
        )
        scheduler = NoamOpt(self.hidden_size, 1, 4000, opt)
        return [opt], [scheduler]
