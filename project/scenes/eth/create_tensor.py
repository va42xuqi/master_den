import config
import os 
from custom_eth_transformation import ETHTransform
from dataset import CustomETHDataloader
from project.utils import (
    CheckTrajectory,
    VelocityTransformation,
    LastTransformation,
    FixValues,
    MoveTrajectories,
    Compose,
)


def make_smalls(fast_dev_run=False):
    transform = Compose(
        [
            CheckTrajectory(x_and_y=True, search_index=1),
            VelocityTransformation(max_velocity=-1, unit="km/h"),
            FixValues(40, 0, "greaterThan", [0, 1]),
            FixValues(-40, 0, "lessThan", [0, 1]),
            MoveTrajectories(x_and_y=True, search_index=0, move_to=0),
            ETHTransform(),
            LastTransformation(delete_features=[2, 5]),
        ]
    )

    if fast_dev_run:
        d_len = config.BATCH_SIZE * 20
    else:
        d_len = -1

    data_path = config.get_load_path()
    if config.TARGET_DATA == "HOTEL":
        game = os.path.join(data_path, "seq_hotel", "obsmat.csv")
    if config.TARGET_DATA == "ETH":
        game = os.path.join(data_path, "seq_eth", "obsmat.csv")
    dataloader = CustomETHDataloader(
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

    if not fast_dev_run:
        dataloader.store(name=config.get_name(), file=config.get_store_path())

def cat(dataloaders, store_name=None):
    base = CustomETHDataloader(
        name=f"{dataloaders[0]}_{config.OBJECT_AMOUNT}OA_{config.STEPS_IN}SI_{config.STEPS_OUT}SO",
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE,
    )

    for dataloader in dataloaders[1:]:
        if dataloader is None:
            continue

        dataloader = f"{dataloader}_{config.OBJECT_AMOUNT}OA_{config.STEPS_IN}SI_{config.STEPS_OUT}SO"
        base = base.cat(
            CustomETHDataloader(
                name=dataloader,
                batch_size=config.BATCH_SIZE,
                shuffle=config.SHUFFLE,
            )
        )
    if store_name is not None:
        store_name = f"{store_name}_{config.OBJECT_AMOUNT}OA_{config.STEPS_IN}SI_{config.STEPS_OUT}SO"
        base.store(store_name, do_exit=False)
    return base


if __name__ == "__main__":
    """
    This script is used to create the tensor data for the ETH dataset.
    fast_dev_run: Set to True if you want to test the script.
    make_smalls: Set to True if you want to create the tensors.
    """
    ###### CONFIG ######
    # 0: False, 1: True
    fast_dev_run = 0
    do_make_smalls = 1
    ####################

    if do_make_smalls:
        make_smalls(fast_dev_run=fast_dev_run)
