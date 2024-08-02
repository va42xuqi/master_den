import numpy as np
from scipy import interpolate

from project.utils import (
    CustomDataloader,
    CustomDatasetPrepare,
    FinalDataset,
    ReadFromCSV,
)
from project.scenes.nba import config


class CustomFinalDataset(FinalDataset):

    def __init__(self, dataset, start, length, isTest=False):
        super().__init__(dataset, start, length)
        self.object_amount = config.OBJECT_AMOUNT

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        return batch

    def __len__(self):
        return super().__len__()


class CustomNBADataloader(CustomDataloader):

    def __init__(
        self,
        name=None,
        data_dir_list=None,
        steps_in=0,
        steps_out=0,
        batch_size=0,
        min_sequence_length=0,
        shuffle=True,
        transform=None,
        split=None,
        num_workers=0,
        seed=-1,
        length=-1,
        scene=None,
        path=None,
    ):
        super().__init__(
            batch_size,
            shuffle,
            name,
            data_dir_list,
            steps_in,
            steps_out,
            min_sequence_length,
            transform,
            split,
            num_workers,
            seed,
            length,
            scene=scene,
            is_hyperlink=False,
            load_path=path,
        )

    def get_dataset(self):
        return CustomNBADataset(
            self.data_dir_list,
            self.steps_in,
            self.steps_out,
            self.min_sequence_length,
            self.transform,
            self.split,
            self.seed,
            self.length,
        )


class CustomNBADataset(CustomDatasetPrepare):
    def __init__(
        self,
        data_dir_list,
        steps_in,
        steps_out,
        min_sequence_length,
        transform,
        split,
        seed,
        length,
    ):
        super().__init__(
            data_dir_list,
            steps_in,
            steps_out,
            min_sequence_length,
            transform,
            split,
            seed,
            length,
        )

    def get_read_from_csv(
        self,
        data_dir_list,
        steps_in,
        steps_out,
        min_sequence_length,
        seed=-1,
        length=-1,
        drop_columns=None,
    ):
        return ReadInput(
            data_dir_list,
            steps_in,
            steps_out,
            min_sequence_length,
            seed,
            length,
            drop_columns,
        )

    def get_time_step(self, batch, prediction=True):
        timestep = 0.04
        x, y, _ = batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        if prediction:
            return [timestep] * y.shape[3]  # y
        else:
            return [timestep] * x.shape[3]  # x

    def get_train(self):
        return CustomFinalDataset(self, 0, int(self.split[0] * self.size))

    def get_val(self):
        return CustomFinalDataset(
            self,
            int(self.split[0] * self.size),
            int((self.split[1] - self.split[0]) * self.size),
        )

    def get_test(self):
        return CustomFinalDataset(
            self,
            int(self.split[1] * self.size),
            int((self.split[2] - self.split[1]) * self.size),
        )

    def normalize_position(self, x):
        return x


class ReadInput(ReadFromCSV):
    def __init__(
        self,
        data_dir_list,
        steps_in,
        steps_out,
        min_sequence_length,
        seed=-1,
        length=-1,
        drop_columns=None,
    ):

        self.player_size = 4
        self.target_player = config.TARGET_DATA
        self.object_amount = config.OBJECT_AMOUNT
        self.last_row = {"game_clock": 720.0, "shot_clock": 24.0}
        self.role_dict = {
            "F-G": 0,
            "F": 1,
            "G": 2,
            "C": 3,
            "F-C": 4,
            "G-F": 5,
            "C-F": 6,
        }
        self.last_quarter = "0"
        self.is_right = 0
        self.side = {"left": -1, "right": 1}
        self.check_duplicates = set()
        super().__init__(
            data_dir_list,
            steps_in,
            steps_out,
            min_sequence_length,
            seed,
            length,
            drop_columns,
        )

    def prepare_sequence(self, sequence):
        # Hier interpolieren
        start_time = sequence[0][0][2]
        end_time = sequence[-1][0][2]
        new_time = np.arange(start_time, end_time, -0.04)
        result = np.zeros((len(new_time), sequence.shape[1], sequence.shape[2]))

        # Interpolieren Sie über die erste Dimension für jede Kombination von Indizes in den anderen Dimensionen
        for i in range(sequence.shape[1]):
            for j in range(sequence.shape[2]):
                old_time = sequence[:, i, 2][::-1]  # Reverse the order of sequence
                old_values = sequence[:, i, j][::-1]  # Reverse the order of sequence
                f = interpolate.interp1d(
                    old_time, old_values, kind="linear", fill_value="extrapolate"
                )  # Reverse the order of sequence
                result[:, i, j] = f(new_time)
        return result

    def get_raw_output(self, index_row, row):
        if (
            row["game_clock"] is None
            or self.last_row["game_clock"] <= row["game_clock"]
            or abs(self.last_row["game_clock"] - row["game_clock"]) > 0.08
        ):
            self.last_row["game_clock"] = row["game_clock"]
            # print("Remove: " + str(index_row) + " " + str(self.last_row['game_clock']) + " " + str(row['game_clock']))
            return []
        elif row["shot_clock"] is None:
            return []
        else:
            self.last_row = row
            if row["quarter"] != self.last_quarter:
                if row["x_0"] * 0.3048 > config.WIDTH / 2:
                    self.is_right = 1
                else:
                    self.is_right = -1
                self.last_quarter = row["quarter"]
            # Check duplicates
            if (row["quarter"], row["game_clock"]) in self.check_duplicates:
                return []
            self.check_duplicates.add((row["quarter"], row["game_clock"]))

            cache = (
                []
            )  # For each player one list with x, y, game_clock (you can add more features for players here)
            iterator = range(10)
            if not row["home_team"] == "Los Angeles Lakers":
                iterator = reversed(iterator)
            for i in iterator:
                cache.append(
                    [
                        row[f"x_{i}"] * 0.3048,
                        row[f"y_{i}"] * 0.3048,
                        row["game_clock"],
                        row[f"x_{i}"] * 0.3048,
                        row[f"y_{i}"] * 0.3048,
                        ((1 if i < 5 else -1) * self.is_right + 1) // 2,
                        self.role_dict[row[f"role_{i}"]],
                    ]
                )
            # ball 
            cache.append(
                [
                    row["ball_x"] * 0.3048,
                    row["ball_y"] * 0.3048,
                    row["game_clock"],
                    row["ball_x"] * 0.3048,
                    row["ball_y"] * 0.3048,
                    -1,
                    -1,
                ]
            )
            return cache[: self.object_amount + 1]  # You can add more features here
