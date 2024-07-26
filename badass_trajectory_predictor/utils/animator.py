import os
import sys
import torch
import numpy as np


import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import badass_trajectory_predictor

from ..utils import (
    velocity_vector_to_position_vector,
    position_to_distance,
)

BEGIN_POS = 2


def animate(
    time_step,
    frames,
    dataloader,
    model,
    model_name,
    config,
    file=None,
    frame_size=None,
    multi_object_function=None,
    scene="NBA",
):
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    fig = plt.figure(figsize=(16, 9))

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    axs = [ax1, ax2]

    def gen_corners(min_x_f, min_y_f, max_x_f, max_y_f):
        if max_x_f == -1:
            max_x_f = torch.max(dataloader.dataset.x[:, BEGIN_POS, :]).item()
            min_x_f = torch.min(dataloader.dataset.x[:, BEGIN_POS, :]).item()
        if max_y_f == -1:
            max_y_f = torch.max(dataloader.dataset.x[:, BEGIN_POS + 1, :]).item()
            min_y_f = torch.min(dataloader.dataset.x[:, BEGIN_POS + 1, :]).item()
        if max_x_f < 0:
            min_x_f = max_x_f
            max_x_f = -max_x_f
        if max_y_f < 0:
            min_y_f = max_y_f
            max_y_f = -max_y_f
        return min_x_f, min_y_f, max_x_f, max_y_f

    min_x, min_y, max_x, max_y = gen_corners(0, 0, frame_size[1], frame_size[0])

    x_label_base = f"Frame: "

    file = "drawing.png"
    if file is not None:
        file_path = os.path.dirname(os.path.realpath(__file__))
        img_path = os.path.join(file_path, "..", "scenes", scene.lower(), file)
    else:
        img_path = None

    dataset = dataloader.dataset.get_test()
    data_len = len(dataset)

    # Preprocess data
    all_x, all_y, all_y_hat = [], [], []
    for i in range(data_len):
        # verbose
        print(f"\r[Animation] Preprocessing data: {i / data_len * 100:.2f}%", end="")
        batch = dataset.__getitem__(i)
        with torch.no_grad():
            _, _, _, y_hat, _, x, y, _ = model.test_step(batch, -1)
        all_x.append(x)
        all_y.append(y)
        all_y_hat.append(y_hat)

    print("\n")

    all_x = torch.cat(all_x, dim=0).detach().cpu()
    all_y = torch.cat(all_y, dim=0).detach().cpu()
    all_y_hat = torch.cat(all_y_hat, dim=0).detach().cpu()

    input_plot = velocity_vector_to_position_vector(
        all_x,
        [time_step] * all_x.size(3),
        all_x[:, :, 2, 0],
        all_x[:, :, 3, 0],
    )

    output_plot = velocity_vector_to_position_vector(
        all_y_hat,
        [time_step] * all_y_hat.size(2),
        input_plot[:, 0, 0, -1],
        input_plot[:, 0, 1, -1],
    )

    target_plot = velocity_vector_to_position_vector(
        all_y,
        [time_step] * all_y.size(3),
        input_plot[:, :, 0, -1],
        input_plot[:, :, 1, -1],
    )

    output_plot = np.concatenate((input_plot[:, 0, :, -1:], output_plot), axis=2)
    target_plot = np.concatenate((input_plot[..., -1:], target_plot), axis=3)

    distance = position_to_distance(output_plot, target_plot[:, 0], axis=1)
    # add distance of 0 for the first frame
    distance = np.concatenate((np.zeros((distance.shape[0], 1)), distance), axis=1)

    if img_path is not None:
        if scene == "NBA":
            arr = plt.imread(img_path)
        if scene == "SOC":
            from PIL import Image

            img = Image.open(img_path).convert("L")
            arr = 1 - np.asarray(img)

    def update(
        i,
        axs=axs,
        distance=distance,
        input_plot=input_plot,
        target_plot=target_plot,
        output_plot=output_plot,
    ):
        print(f"\r[Animation] Creating frame: {i / frames * 100:.2f}%", end="")
        sys.stdout.flush()

        axs[0].clear()
        axs[1].clear()

        for j in range(0, input_plot.shape[1]):
            multi_object_function(axs[0], j, input_plot[i], target_plot[i], rest=None)

        if img_path is not None:
            ax1.imshow(
                arr, cmap="gray", vmin=0, vmax=255, extent=[min_x, max_x, min_y, max_y]
            )

        # tar = [target_plot[i, 0, 0, :], target_plot[i, 0, 1, :]]
        out = [output_plot[i, 0, :], output_plot[i, 1, :]]
        # inp = [input_plot[i, 0, 0, :], input_plot[i, 0, 1, :]]
        dis = distance[i]
        col = ["darkturquoise", "darkorange", "grey"]
        range_x = np.arange(1, len(dis) + 1)

        # axs[0].plot(tar[0], tar[1], label="Target", color=col[0])
        axs[0].plot(out[0], out[1], label="Estimated", color=col[1])
        # axs[0].plot(inp[0], inp[1], label="Target Input", color=col[2])
        axs[0].set_xlabel(x_label_base + str(i))
        axs[0].set_xlim((min_x, max_x))
        axs[0].set_ylim((min_y, max_y))
        axs[0].set_title(f"Test loss")
        axs[0].legend()
        # Remove x and y ticks
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        # Remove the border (spines)
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        axs[0].spines["left"].set_visible(False)
        axs[0].spines["bottom"].set_visible(False)

        axs[1].barh([str(0.04 * k) for k in range_x], dis)  # Show time in seconds
        axs[1].set_xlabel(f"Euclid-Distance in meter")
        axs[1].set_xlim((0, np.max(distance)))
        axs[1].set_ylabel("Time in seconds")  # Change ylabel to Time (s)
        axs[1].invert_yaxis()
        axs[1].xaxis.grid(True, linestyle="--", which="major", color=col[2], alpha=0.25)
        axs[1].set_title("Euclid-Distance over step")

        # Set y ticks
        num_ticks = 10  # Increase number of ticks to 10
        step_size = len(dis) // (num_ticks - 1)
        y_ticks = [str(0.04 * k) for k in range(1, len(dis) + 1, step_size)]
        axs[1].set_yticks(y_ticks)

        plt.tight_layout()
        print(f"\r[Animation] Creating frame: {(i+1) / frames * 100:.2f}%", end="")

    frames = len(distance) if frames == -1 else frames
    frames = min(frames, len(distance))

    fun = animation.FuncAnimation
    anim = fun(fig, update, frames=frames, interval=500, repeat=False)

    root_path = os.path.dirname(badass_trajectory_predictor.__file__)
    path = os.path.join(root_path, "..", "plots")
    anim.save(
        os.path.join(path, scene, f"animation_{model_name}.gif"),
        writer="pillow",
        fps=1 / time_step / 3,
    )


def draw(
    model,
    model_name,
    dataloader,
    config,
    device="cpu" if not torch.cuda.is_available() else "cuda",
    scene="NBA",
    frames=-1,
):
    model.eval()

    frame_size = config.frame_size

    animate(
        config=config,
        model=model,
        model_name=model_name,
        dataloader=dataloader,
        time_step=0.04,
        frames=frames,
        frame_size=frame_size,
        multi_object_function=config.func_color,
        scene=scene,
    )
