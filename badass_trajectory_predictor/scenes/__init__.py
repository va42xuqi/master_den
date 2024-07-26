"""
This file is used to import the configuration files of the different environments.
"""

from .car import get_dataloader as get_dataloader_car
from .eth import get_dataloader as get_dataloader_eth
from .nba import get_dataloader as get_dataloader_nba, CustomNBATransformation, CustomNBADataloader
from .soc import get_dataloader as get_dataloader_soc, CustomSoccerTransformation, CustomSoccerDataloader
from .car import CustomArgoTransformation, CustomCARDataloader

