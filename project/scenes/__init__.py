"""
This file is used to import the configuration files of the different environments.
"""

from .nba import get_dataloader as get_dataloader_nba, CustomNBATransformation, CustomNBADataloader
from .soc import get_dataloader as get_dataloader_soc, CustomSoccerTransformation, CustomSoccerDataloader

