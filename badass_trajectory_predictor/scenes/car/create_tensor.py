import random
import glob
import os

import config
from badass_trajectory_predictor.scenes import (
    CustomArgoTransformation,
    CustomCARDataloader,
)
from badass_trajectory_predictor.utils import Compose


def create_tensor(fast_dev_run=False):
    transform = Compose(
        [
            CustomArgoTransformation(cut=4, track_id=4, focal_track_id=6, cut_pos=6),
        ]
    )

    data_path = get_all_files("argoverse", 1 if fast_dev_run else 16000, ["train"])

    counter = 0
    store_path = rf"{os.path.dirname(__file__)}/data/tensor/"
    for data in data_path:
        data = [data]
        dataloader = CustomCARDataloader(
            data_dir_list=data,
            steps_in=config.STEPS_IN,
            steps_out=config.STEPS_OUT,
            batch_size=-1,  # Flag
            min_sequence_length=config.MIN_SEQUENCE_LENGTH,
            shuffle=config.SHUFFLE,
            num_workers=0,
            split=None,
            transform=transform,
            seed=10,
            length=-1,
        )

        if not fast_dev_run:
            os.makedirs(
                store_path
                + f"{config.TARGET_DATA}_{config.OBJECT_AMOUNT}OA_{config.STEPS_IN}SI_{config.STEPS_OUT}SO",
                exist_ok=True,
            )
            dataloader.store(
                store_path
                + f"{config.TARGET_DATA}_{config.OBJECT_AMOUNT}OA_{config.STEPS_IN}SI_{config.STEPS_OUT}SO/{counter}",
                do_exit=False,
            )
            counter += 1
            print(f"Counter: {counter}")


def download():
    import requests

    url = "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"
    response = requests.get(url, stream=True)

    # Definieren Sie den Pfad, wo die Datei gespeichert werden soll
    save_path = rf"{os.path.dirname(__file__)}/data/argoverse/trainTar/train.tar"
    print("Storing at: ", save_path)

    # Überprüfen Sie, ob die Anforderung erfolgreich war
    if response.status_code == 200:
        # Stellen Sie sicher, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Öffnen Sie die Datei im Schreibmodus als Binärdatei
        count = 0
        with open(save_path, "wb") as file:
            # Schreiben Sie den Inhalt der Antwort in die Datei
            for chunk in response.iter_content(chunk_size=8192):
                count += 1
                if count % 10000 == 0:
                    file_size_mb = file.tell() / 1000 / 1000
                    if file_size_mb // 1000 == 0:
                        print(f"\rDownloaded {file_size_mb:.2f} MB           ", end="")
                    else:
                        print(
                            f"\rDownloaded {file_size_mb / 1000:.2f} GB           ",
                            end="",
                        )
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
    else:
        print(f"Anforderung fehlgeschlagen mit Statuscode {response.status_code}")


def extract():
    import tarfile

    # Definieren Sie den Pfad zur tar-Datei
    tar_file_path = rf"{os.path.dirname(__file__)}/data/argoverse/trainTar/train.tar"

    # Definieren Sie den Pfad, wo die Dateien entpackt werden sollen
    extract_path = rf"{os.path.dirname(__file__)}/data/argoverse/"

    # Öffnen Sie die tar-Datei im Lese-Modus
    with tarfile.open(tar_file_path, "r") as tar:
        # Entpacken Sie alle Dateien im tar-Archiv
        tar.extractall(path=extract_path)


def get_all_files(name, size=None, file=None):
    if file is None:
        file = ["train", "val", "test"]
    files = []
    for f in file:
        files += glob.glob(f"{os.path.dirname(__file__)}/data/{name}/{f}/*")

    result = []
    counter = size if size is not None else len(files)
    random.seed(1)
    random.shuffle(files)
    for folder in files:
        for file in glob.glob(folder + "/*"):
            if file.endswith(".parquet"):
                result.append(os.path.normpath(file))
                counter -= 1
                if counter == 0:
                    return result
    return result


if __name__ == "__main__":
    print("Start creating tensor")

    ###### CONFIG ######
    # 0: False, 1: True
    fast_dev_run = 0
    do_make_smalls = 1
    do_download = 0
    do_extract = 0
    ####################

    if do_download:
        download()
    if do_extract:
        extract()
    if do_make_smalls:
        create_tensor(fast_dev_run)

    print("Ende")
