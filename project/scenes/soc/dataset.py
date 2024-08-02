import numpy as np
import project.utils.dataset_base as dataset_base
from project.scenes.soc import config
import pandas as pd
import torch


class CustomFinalDataset(dataset_base.FinalDataset):

    def __init__(self, dataset, start, length, isTest=False):
        super().__init__(dataset, start, length)
        self.object_amount = config.OBJECT_AMOUNT

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        return batch


class CustomSoccerDataloader(dataset_base.CustomDataloader):

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
        return CustomSoccerDataset(
            self.data_dir_list,
            self.steps_in,
            self.steps_out,
            self.min_sequence_length,
            self.transform,
            self.split,
            self.seed,
            self.length,
        )


class CustomSoccerDataset(dataset_base.CustomDatasetPrepare):
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


class ReadInput(dataset_base.ReadFromCSV):
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

        self.object_amount = config.OBJECT_AMOUNT
        self.array_width = self.object_amount
        self.ball_file = data_dir_list[0].replace("Player", "Ball")
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

    def prepare_data(self, data_csv):
        array_width = self.object_amount * 7

        # load ball file
        ball = pd.read_csv(self.ball_file)

        columns = []
        for i in range(self.object_amount):
            columns.append("frame_" + str(i))
            columns.append("M_" + str(i))
            columns.append("pp_" + str(i))
            columns.append("team_" + str(i))
            columns.append("x_" + str(i))
            columns.append("y_" + str(i))
            columns.append("time_" + str(i))

        result = np.zeros((len(data_csv.groupby("T")), array_width))
        # sort data_csv first by T and then by PersonId (meaning that the first row of each group is the target player)
        data_csv = data_csv.sort_values(by=["T", "PersonId"])
        lol = 0
        for (_, event), (_, ball_event) in zip(
            data_csv.groupby("M"), ball.groupby("M")
        ):
            first_T = event["T"].iloc[0]
            for (_, row), (_, ball_row) in zip(
                event.groupby("T"), ball_event.groupby("T")
            ):
                if len(row) != 22:
                    result[lol, 6] = 0
                    lol += 1
                    continue
                for j, (_, r) in enumerate(row.iterrows()):
                    result[lol, j * 7] = r["N"]
                    result[lol, j * 7 + 1] = r["M"]
                    result[lol, j * 7 + 2] = int(r["PlayingPosition"], 36)
                    result[lol, j * 7 + 3] = int(r["TeamId"].split("-")[2], 36)
                    result[lol, j * 7 + 4] = r["X"]
                    result[lol, j * 7 + 5] = r["Y"]
                    result[lol, j * 7 + 6] = (r["T"] - first_T) / 1_000
                # make ball
                if self.object_amount == 23:
                    if len(ball_row) != 1:
                        result[lol, 6] = 0
                    b = ball_row.iloc[0]
                    k = len(row) * 7
                    result[lol, k] = b["N"]
                    result[lol, k + 1] = b["M"]
                    result[lol, k + 2] = -1
                    result[lol, k + 3] = -int(b["TeamBallPossession"].split("-")[2], 36)
                    result[lol, k + 4] = b["X"]
                    result[lol, k + 5] = b["Y"]
                    result[lol, k + 6] = (
                        b["T"] - first_T
                    ) / 1_000  # Cheat for VeloTransformation
                lol += 1

        return pd.DataFrame(result, columns=columns)

    def file_read(self):
        self.check_duplicates = set()

    def get_raw_output(self, index_row, row):

        if (row[f"time_0"], row[f"x_0"], row[f"x_0"]) in self.check_duplicates:
            print(f"Duplicate: {row[f'time_0']}")
            return []
        self.check_duplicates.add(row[f"time_0"])

        # Check if you find an element
        cache = []
        for i in range(self.object_amount):
            if row[f"time_{i}"] == 0:
                return []
            cache.append(
                [
                    row[f"x_{i}"],
                    row[f"y_{i}"],
                    row[f"time_{i}"],
                    row[f"x_{i}"],
                    row[f"y_{i}"],
                    row[f"team_{i}"],
                    row[f"pp_{i}"],  # Playing Position
                ]
            )
        return cache
