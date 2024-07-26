import torch
import torch.nn as nn

from badass_trajectory_predictor.models.model import Model


# Shared Preprocessing LSTM module for neighbor objects
class SharedPreprocessingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SharedPreprocessingLSTM, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]


# Target LSTM module
class TargetLSTM(nn.Module):
    def __init__(self, target_size, hidden_size, output_size=2):
        super(TargetLSTM, self).__init__()
        self.lstm = nn.LSTM(target_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x.unsqueeze(1), hidden)
        return self.linear(lstm_out[:, -1]), hidden


# Lightning module
class SupLSTM(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_size = 2
        self.input_size = 2
        self.output_size = 2
        player_size = 10

        self.shared_preprocessing_lstm = SharedPreprocessingLSTM(
            self.input_size + 2, self.hidden_size
        )
        self.target_preprocessing_lstm = SharedPreprocessingLSTM(
            self.target_size + 2, self.hidden_size
        )
        self.combining_linear = nn.Linear(
            self.hidden_size * player_size, self.hidden_size
        )
        self.target_lstm = TargetLSTM(
            self.target_size, self.hidden_size, self.output_size
        )

    def preprocess_objects(self, data):
        num_players, batch_size, seq_len, input_dim = data.size()
        target_data = data[0]
        non_target_data = data[1:]
        non_target_players = non_target_data.reshape(
            batch_size * (num_players - 1), seq_len, input_dim
        )

        # Assuming shared_preprocessing_lstm is the shared LSTM encoder
        nt_hidden_states = self.shared_preprocessing_lstm(non_target_players)
        t_hidden_states = self.target_preprocessing_lstm(target_data)

        nt_hidden_states = nt_hidden_states.reshape(batch_size, -1)
        combined_hidden_states = torch.cat([nt_hidden_states, t_hidden_states], dim=1)

        # TODO: add dropout layer

        # Linear transformation
        transformed_state = self.combining_linear(
            combined_hidden_states.view(batch_size, -1)
        )

        return transformed_state

    def forward(self, data, target=None, teacher_forcing=True):
        batch_size = data[0].size(0)
        outputs = torch.zeros(batch_size, self.pred_len, self.target_size).to(
            data[0].device
        )

        combined_hidden_state = self.preprocess_objects(data)
        h_0 = combined_hidden_state.unsqueeze(0)
        c_0 = torch.zeros_like(h_0)

        input_seq = data[0][:, -1, ..., :2]

        for t in range(self.pred_len):
            output, (h_0, c_0) = self.target_lstm(input_seq, (h_0, c_0))
            outputs[:, t, :] = output

            if teacher_forcing and target is not None:
                input_seq = target[:, t, :]
            else:
                input_seq = output

        return outputs

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.swapaxes(0, 1).swapaxes(2, 3)
        y = y.swapaxes(0, 1).swapaxes(2, 3)

        x = x
        y = y[0, ..., :2]

        output = self(x, y, teacher_forcing=True)
        loss = nn.MSELoss()(y, output)
        # more losses
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
            log_dict,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        FDE, ADE, NL_ADE, y_hat, _ = self.test_step(batch, batch_idx)

        NL_ADE = NL_ADE[NL_ADE > 1e-4].mean()

        # Use log_dict for multiple metrics
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
        x = x.swapaxes(0, 1).swapaxes(2, 3)
        y = y.swapaxes(0, 1).swapaxes(2, 3)

        x = x
        y = y[0, ..., :2]

        y_hat = self(x, y, teacher_forcing=False)

        y_hat = y_hat.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        output_pos, predict_pos = self.get_pos(y, y_hat)

        FDE, ADE, NL_ADE, loss_list = self.calculate_metrics(
            predict_pos,
            output_pos,
            y.cpu(),
        )

        return FDE, ADE, NL_ADE, y_hat, loss_list

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=0.001,
            foreach=True if torch.cuda.is_available() else False,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01,
        )
