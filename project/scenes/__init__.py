"""
This file is used to import the configuration files of the different environments.
"""

from .nba import (
    get_dataloader as get_dataloader_nba,
    CustomNBATransformation,
    CustomNBADataloader,
)
from .soc import (
    get_dataloader as get_dataloader_soc,
    CustomSoccerTransformation,
    CustomSoccerDataloader,
)

from .soc.dataset import CustomSoccerDataset
from .nba.dataset import CustomNBADataset

import torch

torch.serialization.add_safe_globals(
    [
        CustomSoccerDataset,
        CustomNBADataset,
        CustomSoccerTransformation,
        CustomNBATransformation,
    ]
)
