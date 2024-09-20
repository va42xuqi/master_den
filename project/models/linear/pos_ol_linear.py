import torch.nn as nn

from project.models.one_step_model import OneStepModel
import torch


class pos_ol_Linear(OneStepModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(
            self.in_features // 2 * self.history_len,
            self.prediction_len * self.output_size,
        )

    def forward(self, src, statics):
        src, statics = self.preprocess_data((src, statics))
        src = src.view(src.size(0), -1)
        embed = self.linear(src)
        return embed.view(embed.size(0), self.prediction_len, self.output_size)

    def preprocess_data(self, data):
        src, statics = data
        src = src[..., 2:]
        src = torch.cat([src[:, :, : self.num_players], src[:, :, -1:]], dim=2)
        statics = torch.cat([statics[:, : self.num_players], statics[:, -1:]], dim=1)
        basket = (
            self.basket_features.to(src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1, 1, 1)
        )
        basket = basket[..., 2:]
        sign = statics[:, 0, 0] * 2 - 1
        basket * sign.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = torch.cat([src, basket], dim=2)
        return out, statics
