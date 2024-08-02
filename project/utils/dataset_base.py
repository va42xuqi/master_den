import csv
import glob
import math
import random
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
import inspect
import os
import sys


class CustomDataloader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        shuffle,
        name=None,
        data_dir_list=None,
        steps_in=0,
        steps_out=0,
        min_sequence_length=0,
        transform=None,
        split=None,
        num_workers=0,
        seed=-1,
        length=-1,
        scene: str = "NBA",
        is_hyperlink=False,
        load_path=None,
    ):
        super().__init__()
        if name is not None:
            if data_dir_list is not None:
                raise ValueError(
                    "Values data_dir_list is not allowed when loading a dataset"
                )

            if steps_in != 0 or steps_out != 0 or transform is not None:
                raise ValueError(
                    "Values steps_in, steps_out and transform are not allowed when loading a "
                    "dataset"
                )
            if load_path is not None:
                path = load_path
            else:
                # go to root of the project (project)
                path = os.path.join(
                    os.path.dirname(inspect.getfile(self.__class__)),
                    "..",
                    "..",
                    "..",
                    "data",
                    scene,
                )
                if is_hyperlink:
                    path = os.path.realpath(path)
            path: str = os.path.join(path, "tensor", name + ".pt")
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist")
            self.dataset = torch.load(path)
            if split is None:
                split = [0.7, 0.9, 1]
            self.dataset.split = split
            steps_in = self.dataset.steps_in
            steps_out = self.dataset.steps_out
            transform = self.dataset.transform
            print("Dataset loaded from: " + path)
        else:
            self.data_dir_list = data_dir_list
            self.min_sequence_length = min_sequence_length
            self.seed = seed
            self.length = length

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.shuffle = shuffle
        self.split = split
        if split is None:
            self.split = [1, 0, 0]  # no val and test

        if data_dir_list is not None:
            self.dataset = self.get_dataset()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.get_train(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset.get_val(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset.get_test(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_dataset(self):
        raise NotImplementedError

    def store(self, name, do_exit=False, is_hyperlink=False, file=None):
        if file is not None:
            dir_path = file
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))

        symlink_path = os.path.join(dir_path, "tensor")
        if is_hyperlink:
            real_path = os.path.realpath(symlink_path)
        else:
            real_path = symlink_path
        os.makedirs(f"{real_path}", exist_ok=True)
        path = os.path.join(real_path, name + ".pt")
        torch.save(self.dataset, path)
        print("Dataset stored at: " + path)
        if do_exit:
            exit(0)
        return self

    def cat(self, other_dataloader):
        cache_x = torch.add(
            other_dataloader.dataset.index_map_x, self.dataset.x.shape[2]
        )
        cache_y = torch.add(
            other_dataloader.dataset.index_map_y, self.dataset.y.shape[2]
        )
        self.dataset.x = torch.cat((self.dataset.x, other_dataloader.dataset.x), dim=2)
        self.dataset.y = torch.cat((self.dataset.y, other_dataloader.dataset.y), dim=2)
        self.dataset.pos = torch.cat(
            (self.dataset.pos, other_dataloader.dataset.pos), dim=0
        )
        self.dataset.size += other_dataloader.dataset.size
        self.dataset.index_map_x = torch.cat((self.dataset.index_map_x, cache_x), dim=0)
        self.dataset.index_map_y = torch.cat((self.dataset.index_map_y, cache_y), dim=0)
        self.print_info()
        print("Datasets concatenated")
        return self

    def space_shuffle(self, random_state):

        # Find sequences
        seqs_x = []
        seqs_idx_x = []
        seqs_y = []
        seqs_idx_y = []
        last = 0
        last_point = 0
        last_idx = 0
        self.steps_in = -1
        # Finde Sequenzen
        if self.steps_in == -1:
            for i in range(len(self.dataset.index_map_x)):
                if self.dataset.index_map_x[i] != 0:
                    # Aufbau: (start1, end1) ... (startN, endN) (Beginn der Seq. bis Ende)
                    seqs_x.append((last_point, int(self.dataset.index_map_x[i])))
                    # Eigentlich hier nicht so wichtig, weil der Fall einfacher ist als der unten
                    # Hier ist der Index der Sequenz (und nicht der in der Map) aber hier nicht so wichtig
                    seqs_idx_x.append((last_idx, i))
                    last_idx = i
                    last_point = int(self.dataset.index_map_x[i])
            seqs_x.append((last_point, self.dataset.x.shape[2]))
            seqs_idx_x.append((last_idx, len(self.dataset.index_map_x)))
        else:
            for i in range(len(self.dataset.index_map_x)):
                if (
                    last + 1 != int(self.dataset.index_map_x[i])
                    and self.dataset.index_map_x[i] != 0
                ):
                    seqs_x.append(
                        (
                            last_point,
                            int(self.dataset.index_map_x[i - 1])
                            + self.dataset.get_input_shape()[2],
                        )
                    )
                    seqs_idx_x.append((last_idx, i))
                    last_point = int(self.dataset.index_map_x[i])
                    last_idx = i
                last = int(self.dataset.index_map_x[i])
            seqs_x.append((last_point, self.dataset.x.shape[2]))
            seqs_idx_x.append((last_idx, len(self.dataset.index_map_x)))
        last = 0
        last_point = 0
        last_idx = 0
        if self.steps_in == -1:
            # Gleicher Code wie für X oben drüber
            for i in range(len(self.dataset.index_map_y)):
                if self.dataset.index_map_y[i] != 0:
                    seqs_y.append((last_point, int(self.dataset.index_map_y[i])))
                    seqs_idx_y.append((last_idx, i))
                    last_idx = i
                    last_point = int(self.dataset.index_map_y[i])
            seqs_y.append((last_point, self.dataset.y.shape[2]))
            seqs_idx_y.append((last_idx, len(self.dataset.index_map_y)))
            assert (
                seqs_idx_x == seqs_idx_y and seqs_idx_y == seqs_y
            )  # Das muss erfüllt sein (weil Sequenzen sich nicht überlappen)
        else:
            for i in range(len(self.dataset.index_map_y)):
                if (
                    last + 1 != int(self.dataset.index_map_y[i])
                    and self.dataset.index_map_y[i] != 0
                ):
                    seqs_y.append(
                        (
                            last_point,
                            int(self.dataset.index_map_y[i - 1])
                            + self.dataset.get_output_shape()[2],
                        )
                    )
                    seqs_idx_y.append((last_idx, i))
                    last_point = int(self.dataset.index_map_y[i])
                    last_idx = i
                last = int(self.dataset.index_map_y[i])
            seqs_y.append((last_point, self.dataset.y.shape[2]))
            seqs_idx_y.append((last_idx, len(self.dataset.index_map_y)))

        if len(seqs_x) != len(seqs_y):
            raise ValueError("Different amount of sequences in x and y")

        print("Found " + str(len(seqs_x)) + " sequences")
        # Shuffle sequences
        permutation = np.arange(len(seqs_x))
        random.Random(abs(random_state)).shuffle(permutation)

        if random_state < 0:
            print("Reverse shuffling")
            permutation = np.argsort(permutation)

        seqs_x = [seqs_x[i] for i in permutation]
        seqs_idx_x = [seqs_idx_x[i] for i in permutation]
        seqs_y = [seqs_y[i] for i in permutation]
        seqs_idx_y = [seqs_idx_y[i] for i in permutation]

        # Rebuild x and y and start_pos index_map
        new_x = torch.zeros(
            (self.dataset.x.shape[0], self.dataset.x.shape[1], self.dataset.x.shape[2]),
            dtype=self.dataset.x.dtype,
        )
        new_y = torch.zeros(
            (self.dataset.y.shape[0], self.dataset.y.shape[1], self.dataset.y.shape[2]),
            dtype=self.dataset.y.dtype,
        )
        new_pos = torch.zeros(
            (
                self.dataset.pos.shape[0],
                self.dataset.pos.shape[1],
                self.dataset.pos.shape[2],
            ),
            dtype=self.dataset.pos.dtype,
        )
        new_index_map_x = torch.zeros((len(self.dataset.index_map_x)), dtype=torch.int)
        new_index_map_y = torch.zeros((len(self.dataset.index_map_y)), dtype=torch.int)
        # start with x
        last_point = 0
        for item in seqs_x:
            for i in range(item[0], item[1]):
                new_x[:, :, last_point] = self.dataset.x[:, :, i]
                last_point += 1

        map_counter = 0
        last_point = 0
        # Bemerkung: Die indexmaps müssen streng Monoton wachsend sein.
        if self.steps_in == -1:
            for item in seqs_idx_x:
                for i in range(item[0], item[1]):
                    # Die Schleife geht immer einer iteration lang (weil keine Überlappungen)
                    new_pos[last_point] = self.dataset.pos[i]  # Michung duchführen
                    new_index_map_x[last_point] = (
                        map_counter  # Der wird jedes mal mit der Eingefügten SequenzLÄNGE addiert (siehe zwei Zeilen unten)
                    )
                    last_point += 1
                    # Ende der Map Randfall. Ber
                    if item[1] == len(self.dataset.index_map_x):
                        map_counter += (
                            self.dataset.x.shape[2] - self.dataset.index_map_x[item[0]]
                        )
                    else:
                        # Hier wird die Länge der Sequenz hinzugefügt d.h. index map aufbau: [0, len1, len1+len2, len1+len2+len3, ...]
                        # Somit kann man sequenzen in dataset.x finden
                        map_counter += (
                            self.dataset.index_map_x[item[1]]
                            - self.dataset.index_map_x[item[0]]
                        )
        else:
            for item in seqs_idx_x:
                for i in range(item[0], item[1]):
                    new_index_map_x[last_point] = map_counter
                    new_pos[last_point] = self.dataset.pos[i]
                    last_point += 1
                    map_counter += 1
                map_counter += self.dataset.get_input_shape()[2] - 1

        last_point = 0
        for item in seqs_y:
            for i in range(item[0], item[1]):
                new_y[:, :, last_point] = self.dataset.y[:, :, i]
                last_point += 1

        map_counter = 0
        last_point = 0
        if self.steps_in == -1:
            for item in seqs_idx_y:
                for i in range(item[0], item[1]):
                    # Gleicher Code wie bei x oben
                    new_index_map_y[last_point] = map_counter
                    last_point += 1
                    # Ende der Map
                    if item[1] == len(self.dataset.index_map_y):
                        map_counter += (
                            self.dataset.y.shape[2] - self.dataset.index_map_y[item[0]]
                        )
                    else:
                        map_counter += (
                            self.dataset.index_map_y[item[1]]
                            - self.dataset.index_map_y[item[0]]
                        )
        else:
            for item in seqs_idx_y:
                for i in range(item[0], item[1]):
                    new_index_map_y[last_point] = map_counter
                    last_point += 1
                    map_counter += 1
                map_counter += self.dataset.get_output_shape()[2] - 1

        def val_check(a, b, name):
            res = torch.sum(a) - torch.sum(b)
            if res != 0:
                print(
                    "Shuffling rounding error at "
                    + name
                    + " with error of : "
                    + str(res)
                )

        # Small Validation check
        val_check(self.dataset.x, new_x, "x")
        val_check(self.dataset.y, new_y, "y")
        val_check(self.dataset.pos, new_pos, "pos")
        self.validation()

        self.dataset.x = new_x
        self.dataset.y = new_y
        self.dataset.pos = new_pos
        self.dataset.index_map_x = new_index_map_x
        self.dataset.index_map_y = new_index_map_y
        print("Sequences shuffled with seed: " + str(random_state))
        return self

    def validation(self):
        if len(self.dataset.index_map_x) != len(self.dataset.index_map_y):
            print(
                f"Different amount of sequences in x({len(self.dataset.index_map_x)}) and "
                f"y({len(self.dataset.index_map_y)})"
            )
            return
        for i in range(1, len(self.dataset.index_map_x)):
            if self.dataset.index_map_x[i] <= self.dataset.index_map_x[i - 1]:
                print(
                    f"x sequence {i} is not in order ({self.dataset.index_map_x[i]} <= {self.dataset.index_map_x[i - 1]})"
                )
            if self.dataset.index_map_y[i] <= self.dataset.index_map_y[i - 1]:
                print(
                    f"y sequence {i} is not in order ({self.dataset.index_map_y[i]} <= {self.dataset.index_map_y[i - 1]})"
                )
        print("Validation done")

    def print_info(self):
        print("Input shape: " + str(self.dataset.get_input_shape()))
        print("Output shape: " + str(self.dataset.get_output_shape()))
        print(
            "Dataset size: ("
            + str(int(self.split[0] * self.dataset.size))
            + ", "
            + str(int((self.split[1] - self.split[0]) * self.dataset.size))
            + ", "
            + str(int((self.split[2] - self.split[1]) * self.dataset.size))
            + ") = "
            + str(self.dataset.size)
        )

    def get_input_shape(self):
        return self.dataset.get_input_shape()

    def get_output_shape(self):
        return self.dataset.get_output_shape()


class Compose(object):
    def __init__(self, transforms, train=False):
        if transforms is None or len(transforms) == 0:
            raise ValueError("transforms must be a list of transformations")
        for t in transforms:
            if not isinstance(t, BaseTransformation):
                raise ValueError("All transforms must be of type BaseTransformation")
        self.transforms = transforms
        self.train = train

    def set_dataloader(self, dataloader):
        for t in self.transforms:
            if isinstance(t, BaseTransformation):
                t.dataloader = dataloader

    def set_train(self, train):
        self.train = train
        return self

    def to_string(self):
        return "Compose:\n\n " + "\n -> ".join(
            [t.to_string() + "\n" for t in self.transforms]
        )

    def __call__(self, x, y, start_pos):
        if self.train:
            for t in self.transforms:
                values = []
                for i in range(x.shape[0]):
                    values.append(t.forward(x[i], y[i], start_pos[i]))
                x, y, start_pos = zip(*values)
                x = torch.stack(x)
                y = torch.stack(y)
                start_pos = torch.stack(start_pos)
        else:
            for t in self.transforms:
                x, y, start_pos = t.forward(x, y, start_pos)
        return x, y, start_pos


def txt_to_csv(path, split=" ", headers=None):
    with open(f"{path}.txt", "r") as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(split) for line in stripped if line)
        with open(f"{path}.csv", "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=headers)
            writer.writeheader()
            for line in lines:
                writer.writerow({header: value for header, value in zip(headers, line)})


class BaseTransformation:
    def __init__(self):
        super().__init__()
        self.dataloader = None

    def forward(self, x, y, startpos):
        raise NotImplementedError

    def to_string(self):
        return self.__class__.__name__


class ReadFromCSV:

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
        if drop_columns is None:
            drop_columns = []
        self.steps_in = steps_in
        self.steps_out = steps_out
        min_sequence_length = max(min_sequence_length, steps_in + steps_out)
        self.size = 0

        self.current_fade = 0
        self.current_fade_index = 0

        if length <= 0:
            length = math.inf

        self.data = []

        for counter, data_dir in enumerate(data_dir_list):
            print(
                f"\r[Dataloader] Reading file: {counter + 1} / {len(data_dir_list)} ({data_dir}) ({len(self.data)})",
                end="",
            )
            sys.stdout.flush()
            if data_dir.endswith(".csv"):
                data_csv = pd.read_csv(data_dir)
            elif data_dir.endswith(".json"):
                data_csv = pd.read_json(data_dir)
            elif data_dir.endswith(".parquet"):
                data_csv = pd.read_parquet(data_dir)
            else:
                raise ValueError("Unknown file format: " + data_dir)

            if data_csv.shape[0] < data_csv.shape[1]:
                data_csv = data_csv.transpose()
                data_csv = data_csv.drop(drop_columns)
            else:
                data_csv = data_csv.drop(drop_columns, axis=1)
            sequence_collection = []
            sequence = []
            data_csv = self.prepare_data(data_csv)

            if data_csv is None:
                continue

            for index, row in data_csv.iterrows():
                output = self.get_raw_output(index, row)

                if output is None:
                    continue
                if len(output) == 0:
                    if len(sequence) > min_sequence_length:
                        sequence_collection.append(np.array(sequence))
                    sequence = []
                    if self.size > length > 0:
                        break

                elif self.size > length > 0:
                    if len(sequence) > min_sequence_length:
                        sequence_collection.append(np.array(sequence))
                        sequence = []
                    break
                else:
                    sequence.append(output)

            if len(sequence) > min_sequence_length:
                sequence_collection.append(np.array(sequence))
            if len(sequence_collection) != 0:
                self.data.append(sequence_collection)
            else:
                print("\nNo sequences found in: " + data_dir)
            self.file_read()
        for block in self.data:
            for i, sequence in enumerate(block):
                block[i] = self.prepare_sequence(sequence)
                if block[i] is None or len(block[i]) < min_sequence_length:
                    block[i] = None
                else:
                    self.size += len(block[i]) - steps_in - steps_out

        self.data = [
            item for sublist in self.data for item in sublist if item is not None
        ]
        if len(self.data) == 0:
            raise ValueError("No sequences found. Your dataset is empty")
        print("\nThere are " + str(len(self.data)) + " sequences in the dataset")
        if seed > 0:
            random.Random(seed).shuffle(self.data)

    # Optional function to prepare the data
    def prepare_data(self, data_csv):
        return data_csv

    def prepare_sequence(self, sequence):
        return sequence

    def file_read(self):
        return

    def get_raw_output(self, index, row):
        raise NotImplementedError

    def __getitem__(self):

        if self.current_fade >= len(self.data):
            raise ValueError("Fehler im Dataloader getitem")

        x, y = list(), list()
        for i in range(self.steps_in):
            # ignore time
            x.append(
                torch.FloatTensor(
                    self.data[self.current_fade][self.current_fade_index + i]
                )
            )

        for i in range(self.steps_out):
            # ignore time
            y.append(
                torch.FloatTensor(
                    self.data[self.current_fade][
                        self.current_fade_index + self.steps_in + i
                    ]
                )
            )

        self.current_fade_index += 1
        if self.current_fade_index >= (
            len(self.data[self.current_fade]) - self.steps_in - self.steps_out
        ):
            self.current_fade_index = 0
            self.current_fade += 1

        x = torch.stack(x, dim=2)
        y = torch.stack(y, dim=2)
        return x, y

    def __getfade__(self):
        if self.current_fade >= len(self.data):
            raise ValueError("Fehler im Dataloader Fade")

        x = list()
        for i in range(len(self.data[self.current_fade])):
            # ignore time
            x.append(torch.FloatTensor(self.data[self.current_fade][i]))

        self.current_fade += 1

        x = torch.stack(x, dim=2)
        return x, x

    def __len__(self):
        return self.size


class CustomDatasetPrepare:
    def __init__(
        self,
        data_dir_list,
        steps_in,
        steps_out,
        min_sequence_length,
        transform,
        split,
        seed=-1,
        length=-1,
        drop_columns=None,
    ):
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.length = length
        min_sequence_length = max(min_sequence_length, steps_in + steps_out)
        self.transform = transform
        helper = self.get_read_from_csv(
            data_dir_list,
            steps_in,
            steps_out,
            min_sequence_length,
            seed,
            length,
            drop_columns,
        )

        if steps_in == -1:
            self.long_sequence = True
        else:
            self.long_sequence = False

        if helper is None:
            raise ValueError("No files read")

        loop_counter = 0
        dummy_start_pos = torch.zeros((1, 2))
        if self.long_sequence:
            helper.size = len(helper.data)

        if transform is not None:
            transform.set_dataloader(self)
        while transform is not None:
            self.x_test, self.y_test, dummy_start_pos = self.get_hxy(helper)
            if self.x_test is not None and self.y_test is not None:
                break
            loop_counter += 1
            if loop_counter >= len(helper) - 1:
                raise ValueError("transform always returns None")

        helper.current_fade_index = 0
        helper.current_fade = 0
        self.x = []
        self.y = []
        self.split = split
        removed_elements = 0
        i = 0
        map_i_x = -self.x_test.shape[2]
        map_i_y = -self.y_test.shape[2]

        self.index_map_x = torch.zeros((len(helper)), dtype=torch.int)
        self.index_map_y = torch.zeros((len(helper)), dtype=torch.int)
        use_pos = True
        if dummy_start_pos is None:
            self.pos = None
            use_pos = False
        else:
            self.pos = torch.zeros(
                (len(helper), self.x_test.shape[0], dummy_start_pos.shape[1])
            )  # for start position
        self.size = len(helper)

        if self.long_sequence:
            print("Input shape: " + str(self.x_test.shape[:-1]) + " +X")
            print("Output shape: " + str(self.y_test.shape[:-1]) + " +Y")

            for current in range(self.size):
                # return all hx and hy for all helper.__getitem__()
                hx, hy, start_pos = self.get_hxy(helper)

                if hx is None or hy is None:
                    removed_elements += 1
                    continue

                if use_pos:
                    self.pos[i] = start_pos

                for j in range(hx.shape[2]):
                    self.x.append(hx[:, :, j])
                for j in range(hy.shape[2]):
                    self.y.append(hy[:, :, j])
                map_i_x += hx.shape[2]
                map_i_y += hy.shape[2]

                self.index_map_x[i] = map_i_x
                self.index_map_y[i] = map_i_y
                i += 1

                if i % 10 == 0:
                    print(
                        f"\r[Dataloader] Transforming data: {((current / self.size) * 100):.2f}% ({current}/{self.size})",
                        end="",
                    )
                    sys.stdout.flush()

        else:
            print("Input shape: " + str(self.x_test.shape))
            print("Output shape: " + str(self.y_test.shape))
            self.size = len(helper)

            def make_list(make_list, tensor, last):
                result = 0
                for j in range(1, tensor.shape[2]):
                    if (
                        torch.abs(torch.sub(tensor[:, :, :-j], last[:, :, j:])).sum()
                        / (tensor.shape[1] * tensor.shape[2])
                        <= 0.0000001
                    ):
                        result = j
                        for k in range(tensor.shape[2] - j, tensor.shape[2]):
                            make_list.append(tensor[:, :, k])
                        break

                if result == 0:
                    result = tensor.shape[2]
                    for j in range(tensor.shape[2]):
                        make_list.append(tensor[:, :, j])

                return result

            last_x = torch.zeros(
                (self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2])
            )
            last_y = torch.zeros(
                (self.y_test.shape[0], self.y_test.shape[1], self.y_test.shape[2])
            )
            for current in range(self.size):
                # return all hx and hy for all helper.__getitem__()
                hx, hy, start_pos = self.get_hxy(helper)

                if hx is None or hy is None:
                    removed_elements += 1
                    continue

                if use_pos:
                    self.pos[i] = start_pos
                map_i_x += make_list(self.x, hx, last_x)
                map_i_y += make_list(self.y, hy, last_y)

                self.index_map_x[i] = map_i_x
                self.index_map_y[i] = map_i_y
                last_x = hx
                last_y = hy

                i += 1

                if i % 1000 == 0:
                    print(
                        f"\r[Dataloader] Transforming data: {((current / self.size) * 100):.2f}% ({current}/{self.size})",
                        end="",
                    )
                    sys.stdout.flush()

        print(
            f"\r[Dataloader] Transforming data: {100:.2f}% ({self.size}/{self.size}) \n",
            end="",
        )
        if self.size > removed_elements > 0:
            self.size = self.size - removed_elements
            if use_pos:
                self.pos = self.pos[:-removed_elements]
            self.index_map_x = self.index_map_x[:-removed_elements]
            self.index_map_y = self.index_map_y[:-removed_elements]
            if removed_elements != 0:
                print(str(removed_elements) + " Elements removed during transformation")

        self.split[1] = self.split[0] + self.split[1]
        self.split[2] = 1
        self.x = torch.stack(self.x, dim=2)
        self.y = torch.stack(self.y, dim=2)
        print(
            "Dataset size: (Train, Val, Test) ("
            + str(int(self.split[0] * self.size))
            + ", "
            + str(int((self.split[1] - self.split[0]) * self.size))
            + ", "
            + str(int((self.split[2] - self.split[1]) * self.size))
            + ")"
            + " = "
            + str(self.size)
        )

    def get_hxy(self, helper):
        if self.long_sequence:
            hx, hy = helper.__getfade__()
        else:
            hx, hy = helper.__getitem__()
        start_pos = torch.stack((hx[:, 0, 0], hx[:, 1, 0]), dim=1)
        if self.transform is not None:
            return self.transform(hx, hy, start_pos)
        return hx, hy, start_pos

    def __getitem__(self, index):
        start_x = self.index_map_x[index]
        start_y = self.index_map_y[index]
        if self.pos is not None:
            return (
                self.x[:, :, start_x : start_x + self.x_test.shape[2]],
                self.y[:, :, start_y : start_y + self.y_test.shape[2]],
                self.pos[index],
            )
        else:
            return (
                self.x[:, :, start_x : start_x + self.x_test.shape[2]],
                self.y[:, :, start_y : start_y + self.y_test.shape[2]],
                None,
            )

    def __len__(self):
        return self.size

    def get_train(self):
        return FinalDataset(self, 0, int(self.split[0] * self.size))

    def get_val(self):
        return FinalDataset(
            self,
            int(self.split[0] * self.size),
            int((self.split[1] - self.split[0]) * self.size),
        )

    def get_test(self):
        return FinalDataset(
            self,
            int(self.split[1] * self.size),
            int((self.split[2] - self.split[1]) * self.size),
        )

    def get_start_test_index(self):
        return int(self.split[1] * self.size)

    def get_start_val_index(self):
        return int(self.split[0] * self.size)

    def get_input_shape(self):
        return self.x_test.shape

    def get_output_shape(self):
        return self.y_test.shape

    def get_trajectory_info_shape(self):
        return self.pos[0].shape

    def set_input_shape(self, shape):
        self.x_test = torch.zeros(shape)

    def set_output_shape(self, shape):
        self.y_test = torch.zeros(shape)

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
        raise NotImplementedError

    def get_time_step(self, batch, prediction=True):
        raise NotImplementedError


class CustomBigFileDataPrepare:

    def __init__(self, data_dir, steps_in, steps_out, split, shuffle, seed):
        if steps_in < 1:
            raise ValueError("steps_in must be greater than 0")
        if steps_out < 1:
            raise ValueError("steps_out must be greater than 0")
        if split[0] < 0 or split[0] > 1:
            raise ValueError("split[0] must be between 0 and 1")
        if split[1] < 0 or split[1] > 1:
            raise ValueError("split[1] must be between 0 and 1")

        self.steps_in = steps_in
        self.steps_out = steps_out
        self.split = split
        self.split[1] = self.split[0] + self.split[1]
        self.split[2] = 1
        self.size = 0
        if type(data_dir) is not str:
            raise ValueError("data_dir must be a string")
        if not data_dir.endswith("/*"):
            data_dir = data_dir + "/*"
        data_dir_list = glob.glob(data_dir)
        self.pt_files = np.array(data_dir_list)
        self.file_dict = {}
        counter = 0
        pt_file_counter = 0
        for pt_file in data_dir_list:
            dataset = torch.load(pt_file)
            self.size += len(dataset)
            for i in range(len(dataset.index_map_x)):
                self.file_dict[counter] = (pt_file_counter, i)
                counter += 1
            pt_file_counter += 1

        if self.file_dict is {}:
            raise ValueError("No files read")

        print("CustomBigFileDataPrepare size: " + str(self.size))

    def __getitem__(self, index):
        file, intern_index = self.file_dict[index]
        dataset = torch.load(self.pt_files[file])
        return dataset.__getitem__(intern_index)

    def get_dataset(self, index):
        file, intern_index = self.file_dict[index]
        return torch.load(self.pt_files[file])

    def __len__(self):
        return self.size

    def get_train(self):
        return FinalDataset(self, 0, int(self.split[0] * self.size))

    def get_val(self):
        return FinalDataset(
            self,
            int(self.split[0] * self.size),
            int((self.split[1] - self.split[0]) * self.size),
        )

    def get_test(self):
        return FinalDataset(
            self,
            int(self.split[1] * self.size),
            int((self.split[2] - self.split[1]) * self.size),
        )

    def get_input_shape(self):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError


class FinalDataset(Dataset):
    def __init__(self, dataset, offset, size):
        self.size = size
        self.dataset = dataset
        self.offset = offset

    def __getitem__(self, index):
        return self.dataset.__getitem__(index + self.offset)

    def __len__(self):
        return self.size
