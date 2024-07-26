import os
import numpy as np
import glob
import sys
import torch
from lightning.pytorch.callbacks import EarlyStopping

from badass_trajectory_predictor.utils import FixProgressBar


STEPS_IN = 50  # -> 38
STEPS_OUT = 200
MIN_SEQUENCE_LENGTH = 50
SHUFFLE = True
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1
BATCH_SIZE = 64
NUM_WORKERS = 0
TARGET_DATA = "ARGO"  # 'ARGO' or 'WAYMO'
OBJECT_AMOUNT = 1000  # All possible pedestrians + objects in the scene
USE_ALL = False  # True = ALL Datasets, False = Only TARGET_DATA

# ANIMATION
FPS = 25
FRAMES = 1000


def custom_animation(model, dataloader, animation_name, experiment):
    from project.utils.animation import create_trajectories_from_data
    from project.utils.animation import concat_mp4
    from project.CAR.customArgoVisualization import plot_trajectory

    test_offset = dataloader.dataset.get_test().offset
    file_index, _ = dataloader.dataset.file_dict[test_offset]

    counter = 0
    for i in range(file_index, len(dataloader.dataset.pt_files)):
        if counter >= FRAMES:
            break
        dataset = torch.load(dataloader.dataset.pt_files[i])
        scenario_id = dataset.scenario
        dataset = dataset.get_train()
        input_list = []
        target_list = []
        output_list = []
        for index in range(len(dataset)):
            if counter >= FRAMES:
                break
            counter += 1
            x, y, rest = dataset.__getitem__(index)
            # Made on 25 FPS
            input_plot, target_plot, output_plot, _ = create_trajectories_from_data(
                x, y, rest, model, 0.04
            )
            input_list.append(input_plot)
            target_list.append(target_plot)
            output_list.append(output_plot)

        # Experiment is Storage name
        plot_trajectory(
            f"{os.path.dirname(__file__)}/data/argoverse/train/" + scenario_id,
            input_list,
            target_list,
            output_list,
        )
        print(
            f"\r[ArgoVis] Visualizing data: {((counter / FRAMES) * 100):.2f}% ({counter}/{FRAMES})"
        )

    # Concat the videos
    concat_mp4(
        f"{os.path.dirname(__file__)}/animation/cache/",
        f"{os.path.dirname(__file__)}/animation/{experiment}",
        and_delete=True,
    )


if __name__ == "__main__":
    from project.train import train

    # Disable cuda
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train(
        config,
        custom_animation=custom_animation,
        opt_info="Simple",
        fast_dev_run=False,
        do_log=True,
        do_animation=False,
    )


def get_path():
    # get path from this file
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, "..", "..", "..")
    p = os.path.join(p, "data", "car")
    return p


def get_load_path():
    p = get_path()
    p = os.path.join(p, "data_raw")
    if "beegfs" in p:
        p = os.path.realpath(p)
    return p


def get_store_path():
    p = get_path()
    return p


def get_plot_path():
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, "..", "..", "..")
    p = os.path.join(p, "plots", "car")
    return p


ANIMATION_SAVE_PATH = get_plot_path()
TB_PATH = "/CAR/tb_logs"

CALLBACKS = [
    EarlyStopping(
        monitor="val_loss", patience=7, mode="min", verbose=True, min_delta=0.001
    ),
    FixProgressBar(),
]

if USE_ALL:
    name = f"ALL_CAR_{OBJECT_AMOUNT}OA_{STEPS_IN}SI_{STEPS_OUT}SO"
else:
    name = f"{TARGET_DATA}_{OBJECT_AMOUNT}OA_{STEPS_IN}SI_{STEPS_OUT}SO"


def get_dataloader():
    from project.CAR.dataset import CustomCARDataloader
    import badass_trajectory_predictor

    return CustomCARDataloader(
        data_dir_list=f"{os.path.dirname(badass_trajectory_predictor.__file__)}/data/car/data_raw",
        name = f"{config.TARGET_DATA}_{config.OBJECT_AMOUNT}OA_{config.STEPS_IN}SI_{config.STEPS_OUT}SO",
        split=[0.7, 0.2, 0.1],
        steps_in=config.STEPS_IN,
        steps_out=config.STEPS_OUT,
        length=1,
        batch_size=config.BATCH_SIZE,
        scene="car",
    )


def get_model(dataloader):
    from project.models.singleCNNLayer.standard import CNN

    return CNN(dataloader=dataloader, learning_rate=LEARNING_RATE)
