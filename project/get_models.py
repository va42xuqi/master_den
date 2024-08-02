import lightning.pytorch as pl

from .models import *


def load_oll(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    return OneLayerLinear(
        hidden_size=None,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )


def load_tll(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    return TwoLayerLinear(
        hidden_size=1024,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )


def load_oslstm(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = OneStepLSTM(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_oslmu(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = OneStepLMU(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_ostf(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = OneStepTrafo(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )
    return model


def load_osbn(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = OneStepBitNet(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_uni_lstm(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = UniLSTM(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_uni_lmu(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = UniLMU(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_uni_bitnet(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = UniBitNet(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_uni_trafo(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = UniTrafo(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_pos_lstm(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = pos_lstm(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_vel_lstm(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = vel_lstm(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_pos_lmu(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = pos_lmu(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_vel_lmu(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = vel_lmu(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_pos_bitnet(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = pos_BitNet(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_vel_bitnet(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = vel_BitNet(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_pos_trafo(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = pos_trafo(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_vel_trafo(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    model = vel_trafo(
        hidden_size=256,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=1024,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )

    return model


def load_vel_ol_linear(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    return vel_ol_Linear(
        hidden_size=None,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )


def load_pos_ol_linear(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    return pos_ol_Linear(
        hidden_size=None,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )


def load_vel_tl_linear(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    return vel_tl_Linear(
        hidden_size=1024,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )


def load_pos_tl_linear(
    config,
    hist_len,
    pred_len,
    has_ball=True,
    has_goals=True,
    pretrain=False,
    fine_tune=False,
) -> pl.LightningModule:
    return pos_tl_Linear(
        hidden_size=1024,
        history_len=hist_len,
        prediction_len=pred_len,
        num_players=config.OBJECT_AMOUNT,
        dropout=0.2,
        has_ball=has_ball,
        config=config,
        has_goals=has_goals,
        pretrain=pretrain,
        fine_tune=fine_tune,
    )
