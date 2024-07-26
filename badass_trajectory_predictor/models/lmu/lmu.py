import torch.nn as nn

from badass_trajectory_predictor.models.model import Model
from badass_trajectory_predictor.models.lmu.lmu_utils.lmu_layer import LMU, LMUFFT


class LMUModel(Model):
    def __init__(
        self,
        memory_size=64,
        theta=1,
        learn_a=False,
        learn_b=False,
        parallel=False,
        **kwargs
    ):
        super(LMUModel, self).__init__(**kwargs)
        self.state = None
        self.lmu = (
            LMU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                memory_size=memory_size,
                theta=theta,
                learn_a=learn_a,
                learn_b=learn_b,
            )
            if not parallel
            else LMUFFT(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                memory_size=memory_size,
                theta=theta,
                seq_len=self.hist_len,
            )
        )
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, reset_state=True):
        if reset_state:
            self.state = None
        out, self.state = self.lmu(x, self.state)
        out = self.linear(out[:, -self.pred_len :])

        return out
