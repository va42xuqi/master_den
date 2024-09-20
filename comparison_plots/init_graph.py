import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_model(file):
    if "lstm" in file:
        model = "LSTM"
    elif "lmu" in file:
        model = "LMU"
    elif "ostf" in file or "trafo" in file:
        model = "Transformer"
    elif "bitnet" in file:
        model = "BitNet"
    elif "one_layer" in file or "1l" in file:
        model = "One Layer Linear"
    elif "two_layer" in file or "2l" in file:
        model = "Two Layer Linear"
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
    angle_fig.grid(True, which='both', linestyle='--', linewidth=0.5)
    error_fig.grid(True, which='both', linestyle='--', linewidth=0.5)

def create_comp(load_path, excluded_words, included_words):
    # Create subplots
    soccer_angle_fig, soccer_angle_ax = plt.subplots()
    # axis
    soccer_angle_ax.set_xlabel("Time (s)")
    soccer_angle_ax.set_ylabel("Mean Angular Error (degrees)")
    soccer_angle_ax.set_title("Soccer Scene")

    soccer_error_fig, soccer_error_ax = plt.subplots()
    nba_angle_fig, nba_angle_ax = plt.subplots()
    nba_error_fig, nba_error_ax = plt.subplots()

    nba_angle_ax.set_xlabel("Time (s)")
    nba_angle_ax.set_ylabel("Mean Angular Error (degrees)")
    nba_angle_ax.set_title("NBA Scene")

    soccer_error_ax.set_xlabel("Time (s)")
    soccer_error_ax.set_ylabel("Mean Error")
    soccer_error_ax.set_title("Soccer Scene")

    nba_error_ax.set_xlabel("Time (s)")
    nba_error_ax.set_ylabel("Mean Error")
    nba_error_ax.set_title("NBA Scene")

    for file in os.listdir(load_path):
        if all(word not in file for word in excluded_words) and all(word in file for word in included_words):
            data = pd.read_csv(os.path.join(load_path, file))
            model = get_model(file)            
            if "SOC" in file:
                print(file)
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

    # Saving figures
    soccer_angle_fig.savefig("soccer_angle.png")
    soccer_error_fig.savefig("soccer_error.png")
    nba_angle_fig.savefig("nba_angle.png")
    nba_error_fig.savefig("nba_error.png")

load_path = "./benchmark"
excluded_words = ["pretrained", "finetuned", "other", "vel", "pos", "_10_", "_25_", "_100_", "uni"]
included_words = ["50"]

create_comp(load_path, excluded_words, included_words)
