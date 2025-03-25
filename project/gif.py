import warnings
import argparse
from project.get_dataloader import data_loader
import matplotlib.patheffects as pe
import os
from project.utils.callbacks import get_callbacks
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Lazy.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.*")


def load(
    pred_len: int = 10,
    hist_len: int = 10,
    batch: int = 0,
):
    scene = "NBA"
    arch = "no_model"
    has_goals = True
    has_ball = True
    mode = "benchmark"

    data, config = data_loader(
        scene=scene, arch=arch, mode=mode, min_sequence_length=hist_len + pred_len
    )

    names_scene = os.path.join(scene.lower(), arch)
    name = arch + "_" + scene + "_" + mode + "_" + str(hist_len) + "_" + str(pred_len)
    callbacks = get_callbacks(names_scene)

    dataset = data.dataset.get_test()
    batch = dataset.__getitem__(i)
    known_features, _, statics = batch

    player_features = known_features[:5, 2:]

    # NBA court dimensions in meters
    court_length = 28.65
    court_width = 15.24

    # Load NBA court image (ensure correct path)
    arr = plt.imread("scenes/nba/drawing.png")

    fig, ax = plt.subplots(figsize=(15, 8))

    # Set NBA court as background (scaled to real court dimensions)
    ax.imshow(arr, extent=[0, court_length, 0, court_width], alpha=0.8)
    team_colors = ['winter', 'autumn']
    team_labels = ['Team 1', 'Team 2']
    arr = plt.imread("scenes/nba/drawing.png")
    ax.imshow(arr, extent=[0, court_length, 0, court_width], alpha=0.8)


    for player_idx in range(player_features.shape[0]):
        team_idx = 0 if player_idx < 5 else 1
        cmap = plt.get_cmap(team_idx and "autumn" or "winter")
        color = cmap(0.2 * (player_idx % 5))

        x_positions = player_features[player_idx, 0, :] 
        y_positions = player_features[player_idx, 1, :] 
        label = team_labels[team_idx] if player_idx % 5 == 0 else None
        ax.plot(x_positions, y_positions, linewidth=2.5, label=label)

        # Add subtle contour with black edges
        ax.plot(x_positions, y_positions, linewidth=2.5, label=label, 
            path_effects=[pe.withStroke(linewidth=4, foreground='black', alpha=0.4)])

    # Set limits with 0.3m margin around the court
    ax.set_xlim(-0.5, court_length + 0.5)
    ax.set_ylim(-0.5, court_width + 0.5)

    # Remove ticks, labels, and axes
    #ax.legend(loc='upper right', fontsize='large')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.axis('off')

    plt.tight_layout()

    os.makedirs("fulltrajectories", exist_ok=True)
    plt.savefig(f"fulltrajectories/player_trajectories_{i}.png")
    plt.close(fig)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--pred_len", type=int, default=100)
    arg.add_argument("--hist_len", type=int, default=50)
    args = arg.parse_args()
    
    i = 0
    while True:
        try:
            load(
                i,
                args.pred_len,
                args.hist_len,
            )
            print(f"Batch {i} processed!")
            i += 1

        except IndexError:
            print("All batches processed!")
            break  