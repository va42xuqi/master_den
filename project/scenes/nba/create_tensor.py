"""
This script is used to create the tensor data for the NBA dataset.
"""

import os
import random

from project.utils import (
    LastTransformation,
    VelocityTransformation,
    Compose,
    SplitXYTransformation,
)

from data_extraction import unzip_all
from project.scenes import CustomNBATransformation
from project.scenes.nba import config
from project.scenes import CustomNBADataloader


def combine_tensors():
    """
    This function combines the tensors for the NBA dataset.
    """
    game = "TOR@LAL"
    base_name = config.get_name()
    games = config.GAMES

    assert any(game in g for g in games), f"{game} not in {games}"

    train_data = []
    for i in range(len(games)):
        name = f"{base_name}_{games[i].split('_')[0]}"
        path = config.get_store_path()
        if game in name:
            test_data = CustomNBADataloader(
                name=name, path=path, batch_size=config.BATCH_SIZE
            )
        else:
            train_data.append(
                CustomNBADataloader(name=name, path=path, batch_size=config.BATCH_SIZE)
            )

    # Shuffle the dataset
    random.seed(42)
    random.shuffle(train_data)

    if test_data not in train_data:
        print("Test data not in train data")

    for i in range(1, len(train_data)):
        train_data[0].cat(train_data[i])

    train_data = train_data[0]

    train_data.space_shuffle(random_state=42)
    test_data.space_shuffle(random_state=42)

    # save the dataloader
    train_data.store(name=config.get_name() + "_train", file=config.get_store_path())
    test_data.store(name=config.get_name() + "_test", file=config.get_store_path())


def make_small(fast_dev_run=False):
    """
    This function creates the tensors for the NBA dataset.
    fast_dev_run: Set to True if you want to test the script.
    """
    games = config.GAMES
    transform = Compose(
        [
            SplitXYTransformation(value=-1, mode=None),
            VelocityTransformation(max_velocity=40, unit="km/h"),
            CustomNBATransformation(),
            LastTransformation(delete_features=2),
        ]
    )

    if fast_dev_run:
        d_len = config.BATCH_SIZE * 40
    else:
        d_len = -1

    PATH = config.get_load_path()

    for game in games:
        print(f"Creating tensor for {game}")
        print(f"percentage done: {games.index(game)/len(games) * 100:.2f}%")
        path = os.path.join(PATH, game + ".parquet")
        dataloader = CustomNBADataloader(
            data_dir_list=[path],
            steps_in=config.STEPS_IN,
            steps_out=config.STEPS_OUT,
            batch_size=config.BATCH_SIZE,
            min_sequence_length=config.MIN_SEQUENCE_LENGTH,
            shuffle=config.SHUFFLE,
            num_workers=4,
            split=[0.7, 0.2, 0.1],
            transform=transform,
            seed=10,
            length=d_len,
        )

        if not fast_dev_run:
            store_path = config.get_store_path()
            dataloader.store(
                name=f"{config.get_name()}_{game.split('_')[0]}",
                file=store_path,
            )


if __name__ == "__main__":
    """
    This script is used to create the tensor data for the NBA dataset.
    fast_dev_run: Set to True if you want to test the script.
    make_unzip: Set to True if you want to unzip the data as parquet files.
    make_smalls: Set to True if you want to create the tensors.
    combine_tensors: Set to True if you want to combine the tensors.
    """
    ###### CONFIG ######
    # 0: False, 1: True
    fast_dev_run = 0
    make_unzip = 1
    make_smalls = 1
    make_combine = 1
    ####################

    if make_unzip:
        data_folder = config.get_raw_path()
        output_folder = config.get_load_path()
        unzip_all(data_folder=data_folder, output_folder=output_folder)
    if make_smalls:
        make_small(fast_dev_run=fast_dev_run)
    if make_combine:
        combine_tensors()
