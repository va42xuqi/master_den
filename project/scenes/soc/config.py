import os

from lightning.pytorch.callbacks import EarlyStopping

from project.utils import FixProgressBar

STEPS_IN = -1  # -> 38
STEPS_OUT = 1
MIN_SEQUENCE_LENGTH = 50
SHUFFLE = True
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 0
OBJECT_AMOUNT = 22  # All possible pedestrians + objects in the scene
USE_ALL = False  # True = ALL Datasets, False = Only TARGET_DATA
TARGET_IDENTIFIER = "00000F"  # string or None if no
frame_size = [-1, -1]  # [height, width] in meters


def get_path():
    # get path from this file
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, "..", "..", "..")
    p = os.path.join(p, "data", "soc")
    return p


def get_name():
    name = f"{TARGET_IDENTIFIER}"
    return name


def get_load_path():
    p = get_path()
    p = os.path.join(p, "data_raw")
    if "beegfs" in p:
        p = os.path.realpath(p)
    p = os.path.join(p, "soccer-camera")
    return p


def get_store_path():
    p = get_path()
    return p


def get_plot_path():
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, "..", "..", "..")
    p = os.path.join(p, "plots", "eth")
    return p


# ANIMATION
FPS = 25
FRAMES = 2000
FRAME_OFFSET = 0
HEIGHT = 15.24  # 28.65 feet
WIDTH = 28.65  # 50 feet

ANIMATION_IMAGE_PATH = None
ANIMATION_SAVE_PATH = get_plot_path()
TB_PATH = "/ETH/tb_logs"

CALLBACKS = [
    EarlyStopping(
        monitor="val_loss", patience=7, mode="min", verbose=True, min_delta=0.001
    ),
    FixProgressBar(),
]

name = "00000F"


def func_color(axs, i, input_plot, target_plot, rest):
    if i == 0:
        axs.plot(
            target_plot[0, 0, :],
            target_plot[0, 1, :],
            label="Ground Truth",
            color="green",
        )
        axs.plot(
            input_plot[0, 0, :],
            input_plot[0, 1, :],
            label="Target Input",
            color="darkgreen",
        )
    elif i == 1:
        axs.plot(
            input_plot[i, 0, :],
            input_plot[i, 1, :],
            label="Team",
            color="royalblue",
            alpha=0.5,
        )
    elif i == 11:
        axs.plot(
            input_plot[i, 0, :],
            input_plot[i, 1, :],
            label="Enemy",
            color="firebrick",
            alpha=0.5,
        )
    elif i < 11:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], color="royalblue", alpha=0.5)
    elif i < 22:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], color="firebrick", alpha=0.5)
    else:
        axs.plot(
            input_plot[i, 0, :],
            input_plot[i, 1, :],
            label="Ball",
            color="black",
            alpha=0.5,
        )


def get_dataloader():
    raise NotImplementedError


def convert_to_centered(self, data):
    return data


def get_goal_position():  # in DFL the sizes are not normed
    goal_offset = 0
    Width = 105
    return [[-WIDTH / 2 + goal_offset, 0], [WIDTH / 2 - goal_offset, 0]]


def get_dataloader(mode="train", arch="lstm", min_sequence_length=MIN_SEQUENCE_LENGTH):
    from project.scenes.soc.dataset import CustomSoccerDataloader

    return CustomSoccerDataloader(
        split=[0.95, 1, 0] if mode == "train" else [0, 0, 1],
        name=(get_name() + "_train" if mode == "train" else get_name() + "_test"),
        path=get_store_path(),
        batch_size=(
            16
            if arch in ["transformer", "tft", "ostf", "os_bitnet", "gpt"]
            else BATCH_SIZE
        ),
        min_sequence_length=min_sequence_length,
    )
