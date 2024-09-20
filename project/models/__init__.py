"""
This module contains the model classes for the trajectory prediction.
"""

from .attention_based import (
    generator,
    TimeSeriesEmbedding,
    OneStepTrafo,
    OneStepBitNet,
    UniBitNet,
    UniTrafo,
    pos_trafo,
    vel_trafo,
    pos_BitNet,
    vel_BitNet,
)
from .linear import (
    OneLayerLinear,
    TwoLayerLinear,
    pos_ol_Linear,
    vel_ol_Linear,
    pos_tl_Linear,
    vel_tl_Linear,
)
from .lmu import OneStepLMU, UniLMU, pos_lmu, vel_lmu
from .lstm import (
    OneStepLSTM,
    UniLSTM,
    pos_lstm,
    vel_lstm,
)

from .one_step_model import OneStepModel