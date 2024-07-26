import numpy as np
import pandas as pd
import torch
from scipy import interpolate

from badass_trajectory_predictor.utils import dataset_base
from badass_trajectory_predictor.scenes.eth import config


class CustomETHDataloader(dataset_base.CustomDataloader):

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
        is_hyperlink=False,
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
            is_hyperlink=is_hyperlink,
            load_path=path,
        )

    def get_dataset(self):
        return CustomETHDataset(
            self.data_dir_list,
            self.steps_in,
            self.steps_out,
            self.min_sequence_length,
            self.transform,
            self.split,
            self.seed,
            self.length,
        )


class CustomETHDataset(dataset_base.CustomDatasetPrepare):
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
            ["y", "vx", "vy", "vz"],
        )

    def get_time_step(self, batch, prediction=True):
        if prediction:
            return [0.04] * batch[1].shape[3]  # y
        else:
            return [0.04] * batch[0].shape[3]  # x

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


class CustomFinalDataset(dataset_base.FinalDataset):

    def __init__(self, dataset, start, length):
        super().__init__(dataset, start, length)
        self.object_amount = config.OBJECT_AMOUNT
        self.dataset.set_input_shape(
            (
                self.object_amount,
                self.dataset.get_input_shape()[1],
                self.dataset.get_input_shape()[2],
            )
        )
        self.dataset.set_output_shape((1, 2, self.dataset.get_output_shape()[2]))

    def __getitem__(self, index):
        return self.standard(super().__getitem__(index))

    def standard(self, item):
        return item + (torch.full((self.object_amount, 2), 0.5),)

    def sort_to_next_neighbor(self, item):
        x, y, start = item
        distances = torch.sqrt(
            (x[:, 2, -1] - x[0, 2, -1]) ** 2 + (x[:, 3, -1] - x[0, 3, -1]) ** 2
        )
        sorted_indices = torch.argsort(distances)
        return (
            x[sorted_indices],
            y[sorted_indices],
            start[sorted_indices],
            torch.full((self.object_amount, 2), 0.5),
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
        self.current_frame = 4
        self.interpolate_factor = 9
        self.array_width = 0
        self.frame_difference = config.FRAME_DIFFERENCE / 10
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
        trajectories = []
        array_width = len(max(data_csv.groupby("frame").indices.values(), key=len))
        # Sort frame
        # data_csv = data_csv.sort_values(by=['frame'])
        data_csv["frame"] = data_csv["frame"].apply(
            lambda x: x + config.FRAME_OFFSET
        )  # Start at 0

        for data_index in data_csv.groupby("id").indices.values():
            frames = []
            for index in data_index:
                frames.append(data_csv.loc[index].to_numpy())

            # Interpolate to make len(frames) + (len(frames) - 1) * self.interpolate_factor equal to 61
            if len(frames) < 2:
                continue
            x = np.arange(0, len(frames))
            spline = interpolate.CubicSpline(x, frames)
            x_new = np.linspace(
                0,
                len(frames) - 1,
                len(frames) + (len(frames) - 1) * self.interpolate_factor,
            )
            frames = spline(x_new)
            # -1 am Ende hinzufügen
            trajectories.append(
                np.pad(frames, ((0, 1), (0, 0)), "constant", constant_values=-1)
            )

        res = np.zeros(
            (
                round(data_csv["frame"].max() * 1 / self.frame_difference + 2),
                array_width,
                5,
            )
        )
        res_map = np.full(array_width, -1)
        last_slot = 0
        slot = 0
        for counter, sequence in enumerate(trajectories):
            # Bestimme die Startposition
            for i, index in enumerate(res_map):
                if index < sequence[0][0]:
                    slot = i
                    break

            if slot == last_slot and res_map[slot] >= sequence[0][0]:
                print(
                    "No free slot found at row: "
                    + str(counter)
                    + " : "
                    + str(sequence[0])
                )

            current_pos = (1 / self.frame_difference) * sequence[0][0]
            current_pos = round(current_pos)
            for i, frame in enumerate(sequence):
                res[current_pos + i][slot][:4] = frame
            last_slot = slot
            res_map[slot] = sequence[-2][0] + 1

        columns = []
        for i in range(array_width):
            columns.append("frame_" + str(i))
            columns.append("id_" + str(i))
            columns.append("x_" + str(i))
            columns.append("y_" + str(i))
            columns.append("time_" + str(i))
        self.array_width = array_width
        for i in range(len(res)):
            time = 0
            for k in range(array_width):
                if res[i][k][0] != 0 and res[i][k][0] != -1:
                    time = res[i][k][0]
                    break
            for k in range(array_width):
                if res[i][k][0] >= 0:
                    res[i][k][4] = time * (0.04 / self.frame_difference)
        return pd.DataFrame(data=res.reshape(res.shape[0], -1), columns=columns)

    def get_raw_output(self, index_row, row):

        # Check if you find an element
        cache = []
        hit = False

        for i in range(self.array_width):
            cache.append(
                [
                    row[f"x_{i}"],
                    row[f"y_{i}"],
                    row[f"time_{i}"],
                    row[f"x_{i}"],
                    row[f"y_{i}"],
                    row[f"id_{i}"],
                    row[f"frame_{i}"],
                ]
            )
            if row["frame_" + str(i)] != 0:
                hit = True
        if hit:
            return cache
        return []
