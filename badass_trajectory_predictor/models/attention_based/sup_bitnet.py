import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from badass_trajectory_predictor.models.model import Model
from badass_trajectory_predictor.models.attention_based import NoamOpt


def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_enc = pos * angle_rates
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    return pos_enc


class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dims, output_dims, sequence_length, dtype=torch.float32):
        super().__init__()
        self.pos_enc = torch.as_tensor(
            positional_encoding(sequence_length, output_dims), dtype=dtype
        )
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        emb = self.linear(x)
        return emb + self.pos_enc.to(x.device)[: emb.shape[-2]]


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        weight_bin = self.weight.sign()  # Binarize weights to -1 or 1
        output = F.linear(input, weight_bin, self.bias)
        return output


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
    ):
        super(CustomTransformerDecoderLayer, self).__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first=batch_first,
        )

        # Replace the feedforward network's linear layers with BitLinear
        self.linear1 = BitLinear(d_model, dim_feedforward)
        self.linear2 = BitLinear(dim_feedforward, d_model)

        # Optionally, replace other linear layers, like those in multi-head attention
        self.self_attn.out_proj = BitLinear(d_model, d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ):
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TargetTransformer(nn.Module):
    def __init__(
        self,
        target_size,
        hidden_size,
        output_size,
        n_head,
        num_decoder_layer,
        seq_length,
    ):
        super().__init__()
        self.embedding = TimeSeriesEmbedding(target_size, hidden_size, seq_length)
        decoder_layer = CustomTransformerDecoderLayer(
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


class SharedPreprocessingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SharedPreprocessingLSTM, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, h_n = self.gru(x)
        return out


class SupBitNet(Model):
    def __init__(self, n_head=2, num_encoder_layer=2, num_decoder_layer=2, **kwargs):
        super().__init__(**kwargs)
        self.target_size = 2
        self.input_size = 2
        self.output_size = 2
        hidden_size = self.hidden_size
        seq_length = self.hist_len
        pred_len = self.pred_len
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
            pred_len,
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

    def forward(self, data, target=None, teacher_forcing=True):
        batch_size = data[0].size(0)
        outputs = torch.zeros(
            batch_size, self.pred_len, self.target_size, device=data[0].device
        )

        combined_hidden_state = self.preprocess_objects(data)
        input_seq = data[0][:, -1, ..., :2].unsqueeze(1)

        for t in range(self.pred_len):
            output = self.target_transformer(input_seq, combined_hidden_state)
            outputs[:, t, :] = output
            if teacher_forcing and target is not None:
                next_seq = target[:, t, :]
            else:
                next_seq = output
            input_seq = torch.cat([input_seq, next_seq.unsqueeze(1)], dim=1)

        return outputs

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.permute(1, 0, 3, 2)
        y = y.permute(1, 0, 3, 2)[0, ..., :2]

        output = self(x, y, teacher_forcing=True)
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
        FDE, ADE, NL_ADE, y_hat, _ = self.test_step(batch, batch_idx)
        NL_ADE = NL_ADE[NL_ADE > 1e-4].mean()
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

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.permute(1, 0, 3, 2)
        y = y.permute(1, 0, 3, 2)[0, ..., :2]

        y_hat = self(x, None, teacher_forcing=False)
        y_hat = y_hat.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        output_pos, predict_pos = self.get_pos(y, y_hat)
        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos, output_pos, y.cpu()
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
        return [opt]
