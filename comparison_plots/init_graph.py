import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "init_graph"

def get_model(file):
    if "lstm" in file:
        model = "LSTM"
    elif "lmu" in file:
        model = "LMU"
    elif "ostf" in file or "trafo" in file:
        model = "Trafo"
    elif "bitnet" in file:
        model = "BitNet"
    elif "one_layer" in file or "1l" in file:
        model = "1LL"
    elif "two_layer" in file or "2l" in file:
        model = "2LL"
    else:
        assert False, "Invalid Model name"
    return model

def create_single(scene, error_fig, angle_fig, model, data):
    angle_mean = data["angular_mean"] / np.pi * 180
    angle_std = data["angular_std"] / np.pi * 180

    error_mean = data["error_mean"]
    error_std = data["error_std"]

    # Generate time steps (assuming they are implicit in the data length)
    time_steps = np.arange(len(data)) * 0.04  # Multiply x-axis by 0.04 (s)

    # Plot the data with transformed time steps
    angle_fig.plot(time_steps, angle_mean, label=model)
    error_fig.plot(time_steps, error_mean, label=model)

    # Adding grid at tick positions
    angle_fig.grid(True, which='both', linestyle='--', linewidth=0.6)
    error_fig.grid(True, which='both', linestyle='--', linewidth=0.5)

def create_comp(load_path, excluded_words, included_words):
    # Set global figure size
    plt.rcParams['figure.figsize'] = [8, 3.5]

    # Soccer scene plots
    soccer_angle_fig, soccer_angle_ax = plt.subplots(dpi=400)
    soccer_error_fig, soccer_error_ax = plt.subplots(dpi=400)

    # NBA scene plots
    nba_angle_fig, nba_angle_ax = plt.subplots(dpi=400)
    nba_error_fig, nba_error_ax = plt.subplots(dpi=400)

    # Setting axis labels and titles
    for ax, label, _ in [(soccer_angle_ax, "Mean Angular Error (degrees)", "Soccer Scene"),
                             (soccer_error_ax, "Mean Error (m)", "Soccer Scene"),
                             (nba_angle_ax, "Mean Angular Error (degrees)", "NBA Scene"),
                             (nba_error_ax, "Mean Error (m)", "NBA Scene")]:
        ax.set_xlabel(r"Time (s)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)  # Smaller ticks
    plt.subplots_adjust(bottom=0.2)

    for file in os.listdir(load_path):
        if all(word not in file for word in excluded_words) and all(word in file for word in included_words):
            data = pd.read_csv(os.path.join(load_path, file))
            model = get_model(file)            
            if "SOC" in file:
                create_single("SOC", soccer_error_ax, soccer_angle_ax, model, data)

            elif "NBA" in file:
                create_single("NBA", nba_error_ax, nba_angle_ax, model, data)

            else:
                assert "NBA" in file or "SOC" in file, "Invalid Scene name"

    # Adding legends to the plots
    soccer_angle_ax.legend()
    soccer_error_ax.legend()
    nba_angle_ax.legend()
    nba_error_ax.legend()

    # Saving figures with 400 DPI
    if not os.path.exists(path):
        os.makedirs(path)

    soccer_angle_fig.show()
    soccer_error_fig.show()
    nba_angle_fig.show()
    nba_error_fig.show()

# Load path and filter settings
load_path = "./benchmark"
excluded_words = ["pretrained", "finetuned", "other", "vel", "pos", "_10_", "_25_", "_100_", "uni"]
included_words = ["50"]

create_comp(load_path, excluded_words, included_words)
