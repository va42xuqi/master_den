"""
This file contains the function data_loader(scene) which returns the dataloader for the specified scene.
"""

import torch

from project import scenes

import scenes.nba.config as nba_config
import scenes.soc.config as soc_config


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
    elif scene == "SOC":
        data = scenes.get_dataloader_soc(
            mode=mode, arch=arch, min_sequence_length=min_sequence_length
        )
        config = soc_config
    else:
        raise ValueError("Scene must be 'NBA' or 'SOC'")

    data.dataset.x = data.dataset.x.to(device)
    data.dataset.y = data.dataset.y.to(device)
    data.dataset.pos = data.dataset.pos.to(device)

    data.dataset.y_test = torch.zeros(
        (data.dataset.y_test.shape[0], 2, data.dataset.y_test.shape[2])
    )

    return data, config
