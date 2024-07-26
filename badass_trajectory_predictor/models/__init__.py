"""
This module contains the model classes for the trajectory prediction.
"""

from .attention_based import (
    RetNet,
    BitNet,
    Trafo,
    SupTrafo,
    SupBitNet,
    TemporalFusionTransformer,
    generator,
    TimeSeriesEmbedding,
    OneStepTrafo,
    OneStepBitNet,
    GPT,
    UniBitNet,
    UniTrafo,
    pos_trafo,
    vel_trafo,
    pos_BitNet,
    vel_BitNet,
)
from .cnn import SingleLayerCNN
from .constant_velocity import NoModel
from .linear import (
    Linear,
    LinearDO,
    OneLayerLinear,
    TwoLayerLinear,
    pos_ol_Linear,
    vel_ol_Linear,
    pos_tl_Linear,
    vel_tl_Linear,
)
from .lmu import LMUModel, ParallelLMU, SupLMU, OneStepLMU, UniLMU, pos_lmu, vel_lmu
from .lstm import (
    LSTMModel,
    ParallelLSTM,
    SupLSTM,
    OneStepLSTM,
    UniLSTM,
    pos_lstm,
    vel_lstm,
)

from .model import Model
from .one_step_model import OneStepModel
from .parallel_model import ParallelModel, NonTargetPlayer
