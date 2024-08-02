import sys

from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateFinder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


class FixProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


def get_callbacks(filename: str) -> [Callback]:
    early_stopping_callback = EarlyStopping(
        monitor="val/ADE",  # Metric to monitor for early stopping
        patience=20,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,  # Prints early stopping updates
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        # unique name with score
        # filename=filename + "-{val/ADE:.2f}",
        filename=filename,
        monitor="val/ADE",
        mode="min",
        save_top_k=1,  # Save the top 1 best model
        verbose=True,
    )
    lr_finder_callback = FineTuneLearningRateFinder(milestones=(5, 10))
    fpb_callback = FixProgressBar()

    return [early_stopping_callback, checkpoint_callback, fpb_callback]
