import os

import numpy as np

from project.utils import (
    ShuffleTrajectories,
    Compose,
)

STEPS_IN = 100  # -> 38
STEPS_OUT = 1
MIN_SEQUENCE_LENGTH = 150
SHUFFLE = True
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 0
TARGET_DATA = "ETH"  # 'ETH' or 'HOTEL'
OBJECT_AMOUNT = 999  # All possible pedestrians + objects in the scene
TARGET_IDENTIFIER = 0  # Represent the id of the pedestrians


def get_path():
    # get path from this file
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, "..", "..", "..")
    p = os.path.join(p, "data", "eth")
    return p


def get_load_path():
    p = get_path()
    return p

def get_store_path():
    p = get_path()
    p = os.path.join(p, "tensor", f"{TARGET_DATA}.pt")
    return p


def get_name():
    name = f"{TARGET_DATA}"
    return name


def get_dataloader(mode="train", arch="lstm", min_sequence_length=MIN_SEQUENCE_LENGTH):
    from project.scenes.eth.dataset import CustomETHDataloader

    return CustomETHDataloader(
        split=[0.8, 0.1, 0.1], # currently no test pt available
        name='obsmat.csv',
        path=get_load_path(),
        batch_size=(
            16 if arch in ["transformer", "tft", "ostf", "os_bitnet"] else BATCH_SIZE
        ),
        min_sequence_length=min_sequence_length,
    )


def func_color(i, trajectory):
    if i == 0:
        # axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], label='Other pedestrian', color='red', alpha=0.8)
        # axs.plot(target_plot[i, 0, :], target_plot[i, 1, :], color='salmon', alpha=0.5)
        return [(128, 0, 128), 1, "Target"], [(0, 0, 255), 1]
    else:
        # axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], color='red', alpha=0.8)
        # axs.plot(target_plot[i, 0, :], target_plot[i, 1, :], color='salmon', alpha=0.5)
        return [(255, 0, 0), 1, "Other Objects"], [(128, 128, 128), 1]


ETH_DATA = [rf"{os.path.dirname(__file__)}/data/seq_eth/obsmat.csv"]
HOTEL_DATA = [rf"{os.path.dirname(__file__)}/data/seq_hotel/obsmat.csv"]

if TARGET_DATA == 'ETH' or TARGET_DATA == 'ETH_HOTEL':
    path = get_load_path()
    H = np.loadtxt(rf'{os.path.join(path, "seq_eth", "H.txt")}').reshape((3, 3))
    ANIMATION_IMAGE_PATH = '/ETH/animation/videos/seq_eth.avi'
    FRAME_DIFFERENCE = 6
    FRAME_OFFSET = 0
elif TARGET_DATA == 'HOTEL':
    path = get_load_path()
    H = np.loadtxt(rf'{os.path.join(path, "seq_hotel", "H.txt")}').reshape((3, 3))
    ANIMATION_IMAGE_PATH = '/ETH/animation/videos/seq_hotel.avi'
    FRAME_OFFSET = 25  # Bitte nicht hinterfragen
    FRAME_DIFFERENCE = 10


# ANIMATION
FPS = 25
FRAMES = 2000
FRAME_OFFSET = 0
HEIGHT = 15.24  # 28.65 feet
WIDTH = 28.65  # 50 feet
