"""
Temporal Fusion Transformer model
"""

__author__ = "Denis Gosalci"
__copyright__ = "Copyright (c) 2024 Denis Gosalci"
__credits__ = ["Denis Gosalci"]

__license__ = "MIT License"
__version__ = "0.2.0"
__maintainer__ = "Denis Gosalci"
__email__ = "denis.gosalci@fau.de"
__status__ = "Development"

import torch
from torch import nn
import torch.optim as optim

from badass_trajectory_predictor.models.attention_based.transformer_utils.temporalFusionDecoder import (
    TemporalFusionDecoder,
)
from badass_trajectory_predictor.models.attention_based.transformer_utils.helper import (
    VariableSelection,
    GRN,
    AddNorm,
)
from badass_trajectory_predictor.models.attention_based.transformer_utils.quantile_loss import (
    QuantileLoss,
)

import lightning.pytorch as pl


class TemporalFusionTransformer(pl.LightningModule):
    def __init__(
        self,
        dataloader,
        learning_rate,
        hidden_size,
        n_heads,
        dropout,
        quantiles=None,
        pred_length = 25,
        history_length = 50,
    ):
        """
        Initialize the Temporal Fusion Transformer model
        Components:
        - Variable Selection:
            - svs: Static Variable Selection
            - pvs: Past Variable Selection
            - fvs: Future Variable Selection
        - Static Covariate Encoder:
            - sce: Static Covariate Encoder
            - sce_cc: Covariates (cell state)
            - sce_ch: Covariates (hidden state)
            - sce_ce: Covariates (enrichment)
            - sce_cs: Covariates (selection)
        - LSTM Encoder-Decoder:
            - LSTM Encoder
            - LSTM Decoder
        - AddNorm
        - Temporal Fusion Decoder
        - Quantiles for prediction

        :param dataloader: The dataloader
        :param learning_rate: The learning rate
        :param hidden_size: The hidden size
        :param n_heads: The number of heads
        :param dropout: The dropout rate
        :param quantiles: The quantiles for prediction

        :return: None
        """

        super().__init__()
        self.data = dataloader
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]

        self.input_shape = dataloader.dataset.get_input_shape()
        self.input_size = self.input_shape[0] * self.input_shape[1] // 2
        self.output_shape = dataloader.dataset.get_output_shape()
        self.future_size = self.input_size
        self.output_size = self.output_shape[1]
        self.static_size = 20  # is not implemented in the dataloader

        self.max_len = pred_length + history_length
        self.pred_length = pred_length
        self.history_length = history_length

        hs = hidden_size

        self.svs = VariableSelection(self.static_size, hs)
        self.pvs = VariableSelection(self.input_size, hs, hs)
        self.fvs = VariableSelection(self.future_size, hs, hs)

        self.sce_cs = GRN(hs)
        self.sce_ce = GRN(hs)
        self.sce_cc = GRN(hs)
        self.sce_ch = GRN(hs)

        self.lstm_encoder = nn.LSTM(hs, hs, batch_first=True)
        self.lstm_decoder = nn.LSTM(hs, hs, batch_first=True)

        self.addnorm = AddNorm(hs, dropout)

        self.tfd = TemporalFusionDecoder(hs, n_heads=n_heads, dropout=dropout)

        for i, q in enumerate(self.quantiles):
            setattr(self, f"q_{q}", nn.Linear(hs, self.output_size))

        self.loss = QuantileLoss()

    def forward(self, static_features, known_features, future_features=None):
        cs, ce, cc, ch = self.static_preprocessing(static_features)
        preprocess, state = self.past_preprocessing(known_features, cs, cc, ch)

        # If future_features are provided, use them for teacher forcing
        if future_features is not None:
            prediction, state = self.prediction_process(future_features, cs, state)
        else:
            prediction = torch.zeros(known_features.size(0), 0, self.hidden_size).to(
                known_features.device
            )
            state = None

        processed_inputs = torch.cat([preprocess, prediction], dim=1)

        # Use teacher forcing for known future features
        if future_features is not None:
            tgt_mask = (
                torch.tril(torch.ones(self.history_length, self.history_length))
                .to(preprocess.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            y = self.tfd(processed_inputs, ce, mask=tgt_mask)[
                :, -future_features.size(1) :
            ]
            y = self.addnorm(y, prediction)
            outs = [getattr(self, f"q_{q}")(y) for q in self.quantiles]
        else:
            # For non-teacher forcing, predict one step at a time using autoregression
            outs = []
            for i in range(self.pred_length):
                y = self.tfd(processed_inputs, ce)[:, -1:]
                y = self.addnorm(y, torch.zeros_like(y))
                outs = [getattr(self, f"q_{q}")(y) for q in self.quantiles]
                best_out = outs[1]  # Use the median prediction for autoregression
                outs.append(best_out)
                out, state = self.prediction_process(best_out, cs, state)
                processed_inputs = torch.cat([processed_inputs, out], dim=1)
            outs = torch.stack(outs, dim=1)

        return outs

    def training_step(self, batch, batch_idx):
        known_features, _, statics_features = batch

        values = (
            known_features[:, :, :2]
            .reshape(known_features.shape[0], -1, known_features.shape[-1])
            .permute(0, 2, 1)
        )
        self.pred_length = 1
        inputs = values[:, :self.history_length-1]
        known_future = values[:, self.history_length -1 : self.history_length + self.pred_length -1]
        target = values[:, self.history_length : self.history_length + self.pred_length, :2]

        # ground truth is the last time step of the known features plus the future features except the last time step
        statics = statics_features.reshape(statics_features.size(0), -1)

        predictions = self(statics, inputs, known_future)
        loss = self.loss.quantile_loss(target, predictions, self.quantiles)
        self.log("train/q_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        known_features, _, statics_features = batch

        values = (
            known_features[:, :, :2]
            .reshape(known_features.shape[0], -1, known_features.shape[-1])
            .permute(0, 2, 1)
        )
        self.pred_length = 1
        inputs = values[:, :self.history_length-1]
        known_future = values[:, self.history_length -1 : self.history_length + self.pred_length -1]
        target = values[:, self.history_length : self.history_length + self.pred_length, :2]

        # ground truth is the last time step of the known features plus the future features except the last time step
        statics = statics_features.reshape(statics_features.size(0), -1)

        predictions = self(statics, inputs, known_future)
        loss = self.loss.quantile_loss(target, predictions, self.quantiles)
        self.log("val/q_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/FDE", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        known_features, _, statics_features = batch

        values = (
            known_features[:, :, :2]
            .reshape(known_features.shape[0], -1, known_features.shape[-1])
            .permute(0, 2, 1)
        )
        self.pred_length = 1
        inputs = values[:, :self.history_length-1]
        known_future = values[:, self.history_length -1 : self.history_length + self.pred_length -1]
        target = values[:, self.history_length : self.history_length + self.pred_length, :2]

        # ground truth is the last time step of the known features plus the future features except the last time step
        statics = statics_features.reshape(statics_features.size(0), -1)

        predictions = self(statics, inputs, known_future)
        loss = self.loss.quantile_loss(target, predictions, self.quantiles)
        loss = self.loss.quantile_loss(target, predictions, self.quantiles)
        self.log("test/q_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def static_preprocessing(self, x):
        """
        Static preprocessing
        :param x: The input tensor
        :return: the tensor after static preprocessing (cs, ce, cc, ch)
        """
        if x is None:
            return None, None, None, None
        x = self.svs(x)
        cs = self.sce_cs(x)
        ce = self.sce_ce(x)
        cc = self.sce_cc(x)
        ch = self.sce_ch(x)
        return cs, ce, cc, ch

    def past_preprocessing(self, x, cs, cc, ch):
        """
        Past preprocessing
        :param x: The input tensor
        :param cs: Static covariates (selection)
        :param cc: Static covariates (cell state)
        :param ch: Static covariates (hidden state)
        :return: the tensor after past preprocessing
        """
        x = self.pvs(x, cs)
        if cc is None:
            cc = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        if ch is None:
            ch = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        y, state = self.lstm_encoder(x, (cc.unsqueeze(0), ch.unsqueeze(0)))
        x = self.addnorm(x, y)

        return x, state

    def prediction_process(self, x, cs, state):
        """
        Prediction process
        :param x: The input tensor
        :param cs: Static covariates (selection)
        :param state: The state of the LSTM
        :return: the tensor after prediction process
        """
        x = self.fvs(x, cs)
        y, state = self.lstm_decoder(
            x, state
        )  # prediction: 1 time step, training: multiple time steps
        x = self.addnorm(x, y)

        return x, state

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
