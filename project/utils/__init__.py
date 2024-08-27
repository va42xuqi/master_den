"""
This module contains the utility functions and classes used in the project.
"""

from .dataset_base import *
from .custom_transformation import *

import torch

torch.serialization.add_safe_globals(
    [Compose, SplitXYTransformation, VelocityTransformation, LastTransformation]
)
