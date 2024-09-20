from torch.optim.lr_scheduler import LRScheduler


class NoamOpt(LRScheduler):
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        del state_dict["optimizer"]
        self.__dict__.update(state_dict)

    def step(self, closure=None):
        self._step += 1
        rate = self.rate()

        for p in self.optimizer.param_groups:
            p["lr"] = rate

        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):

        if step is None:
            step = self._step

        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()
