"""
This module is responsible for providing the dataloader for the soc environment.
"""

from .config import get_dataloader
from .custom_soccer_transformation import CustomSoccerTransformation
from .dataset import CustomSoccerDataloader