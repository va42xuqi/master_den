"""
This file contains the function data_loader(scene) which returns the dataloader for the specified scene.
"""

import torch

from badass_trajectory_predictor import scenes
from badass_trajectory_predictor.utils import dataset_base

import scenes.nba.config as nba_config
import scenes.eth.config as eth_config
import scenes.soc.config as soc_config
import scenes.car.config as car_config


def data_loader(scene, arch=None, mode="NBA", min_sequence_length=0):
    """
    This function returns the dataloader for the specified scene
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if scene == "NBA":
        data = scenes.get_dataloader_nba(
            mode=mode, arch=arch, min_sequence_length=min_sequence_length
        )
        config = nba_config
    elif scene == "ETH":
        data = scenes.get_dataloader_eth()
        config = eth_config
    elif scene == "SOC":
        data = scenes.get_dataloader_soc(
            mode=mode, arch=arch, min_sequence_length=min_sequence_length
        )
        config = soc_config
    elif scene == "CAR":
        data = scenes.get_dataloader_car()
        config = car_config
    else:
        raise ValueError("Scene must be 'NBA' or 'ETH'")

    data.dataset.x = data.dataset.x.to(device)
    data.dataset.y = data.dataset.y.to(device)
    data.dataset.pos = data.dataset.pos.to(device)

    data.dataset.y_test = torch.zeros(
        (data.dataset.y_test.shape[0], 2, data.dataset.y_test.shape[2])
    )

    return data, config
