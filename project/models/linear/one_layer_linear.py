import torch.nn as nn

from project.models.one_step_model import OneStepModel
import torch


class OneLayerLinear(OneStepModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(
            self.in_features * self.history_len,
            self.prediction_len * self.output_size,
        )

    def forward(self, src, statics):
        src, statics = self.preprocess_data((src, statics))
        src = src.view(src.size(0), -1)
        embed = self.linear(src)
        return embed.view(embed.size(0), self.prediction_len, self.output_size)
