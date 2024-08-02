import numpy as np
import torch
from scipy import interpolate
import pandas as pd
import os

from project.utils import dataset_base
from project.scenes.car import config


class CustomCARDataloader(dataset_base.CustomDataloader):

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
        )

    def get_dataset(self):
        if self.batch_size == -1:
            return CustomCARDataset(
                self.data_dir_list,
                self.steps_in,
                self.steps_out,
                self.min_sequence_length,
                self.transform,
                self.split,
                self.seed,
                self.length,
            )
        else:
            return CustomBigDataDataset(
                self.data_dir_list,
                self.steps_in,
                self.steps_out,
                self.split,
                self.seed,
                self.length,
                self.shuffle,
            )


class CustomBigDataDataset(dataset_base.CustomBigFileDataPrepare):
    def __init__(
        self, data_dir_list, steps_in, steps_out, split, seed=0, length=1, shuffle=True
    ):
        super().__init__(data_dir_list, steps_in, steps_out, split, seed, shuffle)
        self.x_test = torch.zeros((length, 4, steps_in))
        self.input_shape = self.x_test.shape
        self.output_shape = torch.zeros((1, 4, steps_out)).shape
        self.length = length

    def get_time_step(self, batch, prediction=True):
        if prediction:
            return [0.04] * batch[1].shape[3]  # y
        else:
            return [0.04] * batch[0].shape[3]  # x

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape

    def get_train(self):
        return CustomFinalDataset(self, 0, int(self.split[0] * self.size), self.length)

    def get_val(self):
        return CustomFinalDataset(
            self,
            int(self.split[0] * self.size),
            int((self.split[1] - self.split[0]) * self.size),
            self.length,
        )

    def get_test(self):
        return CustomFinalDataset(
            self,
            int(self.split[1] * self.size),
            int((self.split[2] - self.split[1]) * self.size),
            self.length,
        )


class CustomCARDataset(dataset_base.CustomDatasetPrepare):
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
        dir_path = os.path.dirname(data_dir_list[0])
        self.scenario = os.path.basename(dir_path)

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

    def get_train(self):
        return CustomFinalDataset(self, 0, int(self.split[0] * self.size), self.length)


class CustomFinalDataset(dataset_base.FinalDataset):

    def __init__(self, dataset, start, end, length=1):
        super().__init__(dataset, start, end)
        self.object_amount = config.OBJECT_AMOUNT
        self.intern_amount = length

    def __getitem__(self, index):
        x, y, rest = super().__getitem__(index)

        def pad_to_length(tensor):
            if len(tensor) < self.intern_amount:
                zeros = torch.zeros(
                    (self.intern_amount - len(tensor),) + tensor.shape[1:]
                )
                tensor = torch.cat([tensor, zeros])
            return tensor

        x_padded = pad_to_length(x)
        y_padded = pad_to_length(y)
        rest_padded = pad_to_length(rest)

        return (
            x_padded[: self.intern_amount],
            y_padded[: self.intern_amount],
            rest_padded[: self.intern_amount],
        )

    def standard(self, item):
        return item + (torch.full((self.object_amount, 2), 0.5),)


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
        self.interpolate_factor = 5
        self.overwrite_array_width = True
        self.array_width = 64
        super().__init__(
            data_dir_list,
            steps_in,
            steps_out,
            min_sequence_length,
            seed,
            length,
            drop_columns,
        )

    def prepare_data(self, data_csv, array_width=0):
        trajectories = []
        time_index = 3

        if array_width == 0:
            data_csv = data_csv.drop(["observed", "scenario_id", "city"], axis=1)
            # convert String to int
            # Make AV to 0
            data_csv["track_id"] = data_csv["track_id"].apply(
                lambda not_x: 0 if not_x == "AV" else not_x
            )
            data_csv["track_id"] = data_csv["track_id"].astype(int)
            data_csv["focal_track_id"] = data_csv["focal_track_id"].astype(int)
            # Make object_type
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 0 if not_x == "vehicle" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 1 if not_x == "pedestrian" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 2 if not_x == "motorcyclist" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 3 if not_x == "cyclist" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 4 if not_x == "bus" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 5 if not_x == "static" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 6 if not_x == "background" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 7 if not_x == "construction" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 8 if not_x == "riderless_bicycle" else not_x
            )
            data_csv["object_type"] = data_csv["object_type"].apply(
                lambda not_x: 9 if not_x == "unknown" else not_x
            )

            if self.overwrite_array_width:
                self.array_width = (
                    len(max(data_csv.groupby("timestep").indices.values(), key=len)) - 1
                )
        else:
            self.array_width = array_width

        for data_index in data_csv.groupby("track_id").indices.values():
            frames = []
            for index in data_index:
                frames.append(data_csv.loc[index].to_numpy())
                if frames[0][time_index] % 2 != 0:
                    frames[-1][time_index] = frames[-1][time_index] + 0.2

            # Nicht gerade schön, aber bei ungeraden Frames kann nicht sauber interpoliert werden
            if len(frames) % 2 == 0:
                del frames[-1]
            # Interpolate to make len(frames) + (len(frames) - 1) * self.interpolate_factor equal to 61
            if len(frames) < 2:
                continue
            x = np.arange(0, len(frames))
            spline = interpolate.CubicSpline(x, frames)
            x_new = np.linspace(
                0,
                len(frames) - 1,
                len(frames) + round((len(frames) - 1) * (self.interpolate_factor - 1)),
            )
            frames = spline(x_new)
            # -1 am Ende hinzufügen
            trajectories.append(
                np.pad(frames, ((0, 2), (0, 0)), "constant", constant_values=-1)
            )

        # Sort trajectories by timestep
        trajectories = sorted(trajectories, key=lambda x: x[0][time_index])
        res = np.zeros(
            (
                round(len(data_csv.groupby("timestep")) * self.interpolate_factor),
                self.array_width,
                len(data_csv.columns),
            )
        )
        res_map = np.full(self.array_width, -1)
        last_slot = 0
        slot = 0
        for counter, sequence in enumerate(trajectories):
            # Bestimme die Startposition
            for i, index in enumerate(res_map):
                if index < sequence[0][time_index]:
                    slot = i
                    break

            if slot == last_slot and res_map[slot] >= sequence[0][time_index]:
                print(
                    "\nNo free slot found at row: "
                    + str(counter)
                    + " : "
                    + str(self.array_width)
                )
                print("correcting slot")
                return self.prepare_data(data_csv, array_width=self.array_width + 1)
                # return None

            # sequence[0][3] is the start position
            current_pos = sequence[0][3] * self.interpolate_factor
            current_pos = round(current_pos)
            for i, frame in enumerate(sequence):
                res[current_pos + i][slot] = frame
            last_slot = slot
            res_map[slot] = sequence[-3][time_index] + 1

        # Downsample
        res = res[::2]

        columns = []
        dtypes_pd = {}
        for i in range(self.array_width):
            for col in data_csv.columns:
                columns.append(col + "_" + str(i))
                if col == "timestep":
                    dtypes_pd[col + "_" + str(i)] = np.float32
                else:
                    dtypes_pd[col + "_" + str(i)] = data_csv[col].dtype

        # Val check if focal_track_id is the complete sequence
        if len(data_csv.groupby("focal_track_id")) != 1:
            print("focal_track_id is not the complete sequence")
        if (
            len(data_csv.loc[data_csv["track_id"] == data_csv["focal_track_id"][0]])
            != 110
        ):
            print(f"focal_track_id is not the complete sequence")

        flag = True
        while flag:
            for i in res[-1, :, time_index]:
                if i != 0:
                    flag = False
                    break
            if flag:
                res = res[:-1]

        return pd.DataFrame(data=res.reshape(res.shape[0], -1), columns=columns).astype(
            dtypes_pd
        )

    def get_raw_output(self, index_row, row):
        cache = []
        test = -1
        for i in range(self.array_width):
            cache.append(
                [
                    row[f"velocity_x_{i}"],
                    row[f"velocity_y_{i}"],
                    row[f"position_x_{i}"],
                    row[f"position_y_{i}"],
                    row[f"track_id_{i}"],
                    row[f"object_type_{i}"],
                    row[f"focal_track_id_{i}"],
                ]
            )
            test += row[f"track_id_{i}"]

        if test == -1:
            return []
        return cache[: self.object_amount]
