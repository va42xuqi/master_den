from project.models.attention_based import TrafoLayer
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.models.one_step_model import OneStepModel
import numpy as np
from scipy.special import legendre

def relative_values(x, target):
    x = x - target.unsqueeze(2)
    return x

def pos_to_basket(x, basket_positions):
    pos = x.clone()
    dist = basket_positions - pos
    return dist

class CNNTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_sizes, strides, paddings):
        super().__init__()
        layers = []
        current_channels = in_channels
        for out_ch, k, s, p in zip(hidden_channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv1d(
                current_channels, out_ch, 
                kernel_size=k, 
                stride=s, 
                padding=p, 
                padding_mode="zeros"
            ))
            layers.append(nn.ReLU())
            current_channels = out_ch
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x, statics):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        return x


class OneStepTrafo(OneStepModel):
    def __init__(self, n_blocks=8, n_heads=6, ffn_hidden=1024, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(4 * 26, self.hidden_size)
        self.encoder = TrafoLayer(
            n_blocks=n_blocks,
            input_dim=self.hidden_size,
            ffn_hidden=ffn_hidden,
            n_heads=n_heads,
            dropout=self.dropout,
            pos_encoding=nn.Identity(),
            generator=nn.Identity(),
            alibi=True,
        )
        self.cnn = CNNTemporalEncoder(
            in_channels=self.in_features,
            hidden_channels=[64, 128, self.hidden_size],
            kernel_sizes=[7, 5, 5],
            strides=[1, 1, 1],
            paddings=[0, 0, 0],  # no padding
        )

        self.fc_out = nn.Linear(
            self.hidden_size, self.prediction_len * self.output_size
        )
        if self.has_goals:
            basket = torch.tensor(self.get_goal_position())
            basket_vel = torch.zeros_like(basket)
            self.basket_features = torch.cat([basket_vel, basket], dim=-1).repeat(
                self.history_len, 1, 1
            )
        self.state = None

    def forward(self, src, statics):
        out, _ = self.preprocess_data((src, statics))
        out = out.flatten(2, 3)
        out = self.cnn(out, statics)
        out = self.encoder(out)[:, -1]

        src = self.fc_out(out)
        return src.view(src.size(0), self.prediction_len, self.output_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-3)
        # scheduler for the transformer model with warmup steps (regression)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=1134,   # ~3 epochs
            T_mult=1,   # Constant restart interval length
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
        }
    }
