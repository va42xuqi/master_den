from badass_trajectory_predictor.models.attention_based import (
    TimeSeriesEmbedding,
)

from badass_trajectory_predictor.models.attention_based import TrafoLayer
from .transformer_utils import NoamOpt
import torch
import torch.nn as nn

from badass_trajectory_predictor.models.one_step_model import OneStepModel
import random
from badass_trajectory_predictor.scenes.nba.custom_nba_transformation import (
    sliding_transformation,
)
from . import subsequent_mask


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
    cosine_similarity = dot_product / (magnitude_y * magnitude_y_hat)

    # Clamp the cosine similarity to the valid range [-1, 1]
    cosine_similarity = torch.clamp(cosine_similarity, min=-1, max=1)

    # Calculate the angular error between the predicted and ground truth velocity vectors
    angular_error = torch.acos(cosine_similarity)

    return angular_error


def relative_values(x, target):
    x = x - target.unsqueeze(2)
    return x


def pos_to_basket(x, basket_positions):
    pos = x.clone()
    dist = basket_positions - pos
    return dist


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, statics):
        x = x.permute(0, 2, 1)
        x = self.conv(x)

        x = x.permute(0, 2, 1)
        return x


class GPT(OneStepModel):
    def __init__(self, n_blocks=8, n_heads=6, ffn_hidden=1024, **kwargs):
        super().__init__(**kwargs)
        # self.pos_encoder = TimeSeriesEmbedding(
        #    self.in_features,
        #    self.hidden_size,
        #    self.history_len,
        #    dropout=0.0,
        # )

        self.encoder = CNNEncoder(
            in_channels=4,
            out_channels=self.hidden_size,
            kernel_size=13,
            stride=13,
            padding=0,
        )
        self.transformer = TrafoLayer(
            n_blocks=n_blocks,
            input_dim=self.hidden_size,
            ffn_hidden=ffn_hidden,
            n_heads=n_heads,
            dropout=self.dropout,
            pos_encoding=nn.Identity(),
            generator=nn.Identity(),
        )
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
        basket = torch.tensor(self.get_goal_position())
        basket_vel = torch.zeros_like(basket)
        self.basket_features = torch.cat([basket_vel, basket], dim=-1)

    def preprocess_data(self, data):
        src, statics = data
        basket = (
            self.basket_features.to(src.device)
            .unsqueeze(0)
            .repeat(src.size(0), src.size(1), 1, 1)
        )
        src = torch.cat([src, basket], dim=2)
        return src, statics

    def forward(self, src, targets=None, statics=None):
        src_out, _ = self.preprocess_data((src, statics))
        if targets is not None:
            tgt_out, _ = self.preprocess_data((targets, statics))
            src = torch.cat([src_out, tgt_out], dim=1)
        else:
            src = src_out

        src = src.flatten(1, 2)
        encod = self.encoder(src, statics)

        if targets is not None:
            out = []
            inp = encod[:, : self.history_len]
            for i in range(self.prediction_len):
                tmp = self.transformer(inp)
                out.append(tmp[:, -1])
                # random teacher forcer:
                if random.random() < 0.5:
                    inp = torch.cat(
                        [
                            encod[:, : self.history_len + i],
                            encod[:, self.history_len + i].unsqueeze(1),
                        ],
                        dim=1,
                    )
                else:
                    inp = torch.cat(
                        [
                            encod[:, : self.history_len],
                            torch.stack(out, dim=1),
                        ],
                        dim=1,
                    )
            out = torch.stack(out, dim=1)
        else:
            out = []
            for _ in range(self.prediction_len):
                tmp = self.transformer(encod)
                out.append(tmp[:, -1])
                encod = torch.cat([encod, tmp[:, -1:]], dim=1)
            out = torch.stack(out, dim=1)
        y = self.fc_out(out)
        return y

    def step(self, batch, num_batches=10, shuffle=True, teacher_forcing=False):
        known_features, _, statics = batch

        if shuffle:
            known_features[:, : self.num_players] = random_rotate_blocks(
                known_features[:, : self.num_players], dim=1
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
        loss = 0

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

            if teacher_forcing:
                targets = future_features_list[i][:, : self.prediction_len - 1]
            else:
                targets = None
            output = self(inp, targets, statics)
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
        loss = self.step(batch, teacher_forcing=True)[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, x, y = self.step(batch, teacher_forcing=False)

        y = y[:, 0]
        output_pos, predict_pos = self.get_pos(y, output, pred_len=self.prediction_len)

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
            threshold=0.5,
        )

        NL_ADE = NL_ADE[NL_ADE > 0.5].mean()

        self.log("val/FDE", FDE, on_step=True, on_epoch=True)
        self.log("val/ADE", ADE, on_step=True, on_epoch=True)
        self.log("val/NL_ADE", NL_ADE, on_step=True, on_epoch=True)

        return FDE

    def test_step(self, batch, batch_idx, **kwargs):
        loss, output, x, y = self.step(
            batch, shuffle=False, num_batches=-1, teacher_forcing=False
        )

        z = y[:, 0]

        angular_error = get_angular_error(z, output)

        output_pos, predict_pos = self.get_pos(z, output, pred_len=self.prediction_len)

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            z.cpu(),
            threshold=0.5,
        )
        return FDE, ADE, NL_ADE, output, loss_list, x, y, angular_error

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = NoamOpt(
            model_size=self.hidden_size, factor=0.5, warmup=4000, optimizer=optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
