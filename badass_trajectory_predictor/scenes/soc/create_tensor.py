import os

from badass_trajectory_predictor.scenes.soc import config
from badass_trajectory_predictor.scenes import CustomSoccerDataloader

import pandas as pd
from badass_trajectory_predictor.utils import (
    LastTransformation,
    VelocityTransformation,
    Compose,
    SplitXYTransformation,
)
from badass_trajectory_predictor.scenes import CustomSoccerTransformation


def make_small(negative=False, fast_dev_run=False):
    transform = Compose(
        [
            SplitXYTransformation(value=-1, mode=None),
            VelocityTransformation(max_velocity=40, unit="km/h", has_ball=True),
            CustomSoccerTransformation(move="TW"),
            LastTransformation(delete_features=2),
        ]
    )

    if fast_dev_run:
        d_len = config.BATCH_SIZE * 40
    else:
        d_len = -1

    DFL_CLU_00000F = [
        "DFL-MAT-0002UG-Player.csv",
        "DFL-MAT-0002UR-Player.csv",
        "DFL-MAT-0002V1-Player.csv",
        "DFL-MAT-0002VT-Player.csv",
        "DFL-MAT-0002W3-Player.csv",
        "DFL-MAT-0002W8-Player.csv",
        "DFL-MAT-0002WN-Player.csv",
        "DFL-MAT-0002WT-Player.csv",
        "DFL-MAT-0002X3-Player.csv",
        "DFL-MAT-0002X5-Player.csv",
        "DFL-MAT-0002XO-Player.csv",
        "DFL-MAT-0002XP-Player.csv",
        "DFL-MAT-0002XV-Player.csv",
        "DFL-MAT-0002YA-Player.csv",
        "DFL-MAT-0002YE-Player.csv",
        "DFL-MAT-0002YG-Player.csv",
        "DFL-MAT-0002YZ-Player.csv",
        "DFL-MAT-0002Z3-Player.csv",
        "DFL-MAT-0002ZO-Player.csv",
        "DFL-MAT-0002ZV-Player.csv",
        "DFL-MAT-0002ZY-Player.csv",
        "DFL-MAT-000301-Player.csv",
        "DFL-MAT-000304-Player.csv",
        "DFL-MAT-00030P-Player.csv",
        "DFL-MAT-00030V-Player.csv",
        "DFL-MAT-000314-Player.csv",
        "DFL-MAT-000317-Player.csv",
        "DFL-MAT-000319-Player.csv",
        "DFL-MAT-00031L-Player.csv",
        "DFL-MAT-00031M-Player.csv",
        "DFL-MAT-000325-Player.csv",
        "DFL-MAT-00032F-Player.csv",
        "DFL-MAT-00032S-Player.csv",
    ]
    game_list = DFL_CLU_00000F
    split = [0.7, 0.2, 0.1]

    real_path = config.get_load_path()
    print(len(game_list))
    game_list = [os.path.join(real_path, game) for game in game_list]
    for game in game_list:
        dataloader = CustomSoccerDataloader(
            data_dir_list=[game],
            steps_in=config.STEPS_IN,
            steps_out=config.STEPS_OUT,
            batch_size=config.BATCH_SIZE,
            min_sequence_length=config.MIN_SEQUENCE_LENGTH,
            shuffle=config.SHUFFLE,
            num_workers=0,
            split=[0.7, 0.2, 0.1],
            transform=transform,
            seed=10,
            length=d_len,
        )

        game_name = game.split("-")[-2]

        if not fast_dev_run:
            dataloader.store(
                f"00000F_{game_name}",
                file=config.get_store_path(),
                do_exit=False,
            )


def combine_tensors():
    store_path = config.get_store_path()
    path = os.path.join(store_path, "tensor")
    GAMES = os.listdir(path)
    test = ["00000F_0002ZV.pt"]
    train = [g for g in GAMES if g not in ["00000F_0002ZV.pt", "SOCC_train.pt", "SOCC_test.pt"]]
    base = CustomSoccerDataloader(
        name=train[0].split(".")[0],
        path=store_path,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE,
    )
    for game in train[1:]:
        tmp = os.path.join(path, game)
        dataloader = CustomSoccerDataloader(
            name=game.split(".")[0],
            path=store_path,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE,
        )
        base = base.cat(dataloader)
    base.space_shuffle(42)
    base.store(name=config.get_name() + "_train", file=config.get_store_path())

    base = CustomSoccerDataloader(
        name="00000F_0002ZV",
        path=store_path,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE,
    )
    base.space_shuffle(42)
    base.store(name=config.get_name() + "_test", file=config.get_store_path())

def find_matches(team_name):
    real_path = config.get_real_path()
    GAMES = os.listdir(real_path)
    # delete games that does include BALL in the name
    GAMES = [game for game in GAMES if "Ball" not in game]
    result = []
    for game in GAMES:
        df = pd.read_csv(os.path.join(real_path, game))
        if team_name in df["TeamId"].values:
            print(str(set(df["TeamId"].values)) + game)
            result.append(game)

    print(result)


if __name__ == "__main__":
    ###### CONFIG ######
    fast_dev_run = 0
    make_smalls = 0
    make_combine = 1
    negative = 0
    ####################

    if make_smalls:
        make_small(negative=negative, fast_dev_run=fast_dev_run)
    if make_combine:
        combine_tensors()
    print("Ende")
