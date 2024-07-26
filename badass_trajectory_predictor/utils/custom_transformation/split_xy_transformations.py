from badass_trajectory_predictor.utils.dataset_base import BaseTransformation


class SplitXYTransformation(BaseTransformation):

    def __init__(self, value, mode):
        super().__init__()
        self.value = value
        self.mode = mode
        if mode == 'percent':
            assert 0 < value < 1, "Value must be between 0 and 1"
        else:
            self.value = int(value)

    def forward(self, x, y, start_pos):
        # Mischen Sie nur die ersten n Spieler
        if self.mode == 'percent':
            cut = int(x.shape[2] * self.value)
            y = x[:, :, cut:]
            x = x[:, :, :cut]
        else:
            if self.mode == 'step_out':
                self.value = -self.dataloader.steps_out
            y = x[:, :, self.value:]
            x = x[:, :, :self.value]

        return x, y, start_pos

    def to_string(self):
        return f"SplitXYTransformation(value={self.value}, mode={self.mode})"
