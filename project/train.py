import warnings
import argparse
import os
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from project.utils.callbacks import get_callbacks
from project.utils.animator import draw
from project.get_dataloader import *
from project.get_models import *
from project.utils.test_plot import visualize_predictions

# Ignore warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Lazy.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.*")


def load(
    mode: str = "train",
    scene: str = "NBA",
    arch: str = "lstm",
    pred_len: int = 10,
    hist_len: int = 10,
    pretrain: bool = False,
    fine_tune: bool = False,
    fine_tune_scene: str = "SOC",
    logger: str = "wandb",
):
    model_names = [
        "ostf",
        "oslstm",
        "oslmu",
        "one_layer_linear",
        "two_layer_linear",
        "os_bitnet",
        "uni_lstm",
        "uni_lmu",
        "uni_bitnet",
        "uni_trafo",
        "pos_lstm",
        "vel_lstm",
        "pos_lmu",
        "vel_lmu",
        "pos_bitnet",
        "vel_bitnet",
        "pos_trafo",
        "vel_trafo",
        "pos_1l_linear",
        "vel_1l_linear",
        "pos_2l_linear",
        "vel_2l_linear",
    ]
    model_loaders = [
        load_ostf,
        load_oslstm,
        load_oslmu,
        load_oll,
        load_tll,
        load_osbn,
        load_uni_lstm,
        load_uni_lmu,
        load_uni_bitnet,
        load_uni_trafo,
        load_pos_lstm,
        load_vel_lstm,
        load_pos_lmu,
        load_vel_lmu,
        load_pos_bitnet,
        load_vel_bitnet,
        load_pos_trafo,
        load_vel_trafo,
        load_pos_ol_linear,
        load_vel_ol_linear,
        load_pos_tl_linear,
        load_vel_tl_linear,
    ]
    scene = scene.upper()

    if arch not in model_names:
        print(f"Invalid model architecture: {arch}")
        return

    if scene not in ["NBA", "ETH", "SOC", "CAR"]:
        print(f"Invalid scene: {scene}")
        return

    if mode not in ["train", "animate", "benchmark"]:
        print(f"Invalid mode: {mode}")
        return

    i = model_names.index(arch)

    # this will decide if the model is a team game or not (otherwise it is not a team game)
    is_team_game = scene == "NBA" or scene == "SOC"
    has_goals = is_team_game
    has_ball = is_team_game

    data, config = data_loader(
        scene=scene, arch=arch, mode=mode, min_sequence_length=hist_len + pred_len
    )
    model = model_loaders[i](
        config,
        hist_len,
        pred_len,
        has_ball,
        has_goals,
        pretrain,
        fine_tune,
    )

    names_scene = os.path.join(scene.lower(), model_names[i])
    name = arch + "_" + scene + "_" + mode + "_" + str(hist_len) + "_" + str(pred_len)
    if pretrain:
        names_scene += "_pretrain"
        name += "_pretrain"
    if fine_tune:
        names_scene += f"_{fine_tune_scene.lower()}_fine_tune"
        name += f"_{fine_tune_scene.lower()}_fine_tune"
    callbacks = get_callbacks(names_scene)

    def load_checkpoint(model, checkpoint_name):
        checkpoint = torch.load(
            f"checkpoints/{scene.lower()}/{checkpoint_name}",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    checkpoint_name = None
    if fine_tune:
        checkpoint_name = f"{model_names[i]}_pretrain.ckpt"
    if mode == "animate" or mode == "benchmark":
        if pretrain:
            checkpoint_name = f"{model_names[i]}_pretrain.ckpt"
        elif fine_tune:
            checkpoint_name = f"{model_names[i]}_fine_tune.ckpt"
        else:
            checkpoint_name = f"{model_names[i]}.ckpt"

    if checkpoint_name:
        model = load_checkpoint(model, checkpoint_name)

    if logger == "tensorboard":
        logger = TensorBoardLogger("log", name=names_scene)
    elif logger == "wandb":
        logger = WandbLogger(log_model="all", name=name)
    else:  # not implemented or invalid logger
        logger = None

    trainer = pl.Trainer(
        max_epochs=-1,
        num_nodes=1,
        devices=1,
        callbacks=callbacks,
        fast_dev_run=False,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        strategy="auto",
        gradient_clip_val=0.5,
    )

    if fine_tune:
        data, config = data_loader(
            scene=fine_tune_scene,
            arch=arch,
            mode=mode,
            min_sequence_length=hist_len + pred_len,
        )

    if mode == "train" and model_names[i] != "no_model":
        trainer.fit(
            model,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader(),
        )
        trainer.validate(model, data.val_dataloader())

    if mode == "animate":
        draw(
            model,
            model_names[i],
            dataloader=data,
            config=config,
            frames=1000,
            scene=scene,
        )

    if mode == "benchmark":
        visualize_predictions(
            model=model,
            model_name=model_names[i],
            test_dataloader=data.dataset.get_test(),
            num_samples=None,
            scene=scene if not fine_tune else fine_tune_scene,
            pred_len=pred_len,
            pretrained=pretrain,
            fine_tuned=fine_tune,
            hist_len=hist_len,
        )


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--arch", type=str, default="one_layer_linear")
    arg.add_argument("--mode", type=str, default="train")
    arg.add_argument("--scene", type=str, default="SOC")
    arg.add_argument("--pred_len", type=int, default=100)
    arg.add_argument("--hist_len", type=int, default=50)
    arg.add_argument("--pretrain", action="store_true")
    arg.add_argument("--fine_tune", action="store_true")
    arg.add_argument("--fine_tune_scene", type=str, default="NBA")
    arg.add_argument("--seed", type=int, default=42)
    arg.add_argument("--logger", type=str, default="wandb")
    assert arg.parse_args().mode in ["train", "animate", "benchmark"]
    assert arg.parse_args().scene in ["NBA", "ETH", "SOC", "CAR"]
    assert (
        arg.parse_args().pretrain is not True or arg.parse_args().fine_tune is not True
    )
    args = arg.parse_args()
    load(
        args.mode,
        args.scene,
        args.arch,
        args.pred_len,
        args.hist_len,
        args.pretrain,
        args.fine_tune,
        args.fine_tune_scene,
        args.logger,
    )

    # print parameters
    print("Parameters:")
    print("arch: ", args.arch)
    print("mode: ", args.mode)
    print("scene: ", args.scene)
    print("pred_len: ", args.pred_len)
    print("hist_len: ", args.hist_len)
    print("pretrain: ", args.pretrain)
    print("fine_tune: ", args.fine_tune)
    print("fine_tune_scene: ", args.fine_tune_scene)
