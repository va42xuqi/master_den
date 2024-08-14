import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def set_output_file(model_name, scene, file_suffix, hist_len):
    file_name = f"{model_name}_{scene}"
    file_name += file_suffix
    file_name += f"_{hist_len}"
    file_name += ".txt"
    output_dir = "../output_logs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    sys.stdout = open(file_path, "w")
    global original_stdout
    original_stdout = sys.__stdout__

def reset_output():
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def plot_error_angular_error(distances, angles, scene, model_name, file_suffix):
    dir = f"../benchmark/{scene}/{hist}/{model_name}{file_suffix}"
    os.makedirs(dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.plot(angles, distances, "o")
    plt.title("Error of the model")
    plt.xlabel("Angular distance")
    plt.ylabel("Error")
    plt.savefig(f"{dir}/angular_distance_error.png")

    return distances, angles


def plot(
    time_axis,
    error_mean,
    error_var,
    angular_mean,
    angular_var,
    model_name,
    scene,
    file_suffix,
    hist_len,
):
    dir = f"../plots/{scene}/{hist_len}/{model_name}{file_suffix}"
    os.makedirs(dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.errorbar(time_axis, error_mean, yerr=error_var, fmt="o")
    plt.title("Error of the model")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.savefig(f"{dir}/error.png")

    plt.figure(figsize=(10, 10))
    plt.errorbar(time_axis, angular_mean, yerr=angular_var, fmt="o")
    plt.title("Angular Error of the model")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.savefig(f"{dir}/angular_error.png")

    # Save mean and variance of angular error to a file
    dir = f"../benchmark/{scene}/{hist_len}/{model_name}{file_suffix}"
    os.makedirs(dir, exist_ok=True)
    np.save(f"{dir}/error_mean.npy", error_mean)
    np.save(f"{dir}/error_var.npy", error_var)
    np.save(f"{dir}/angular_mean.npy", angular_mean)
    np.save(f"{dir}/angular_var.npy", angular_var)
    return error_mean, error_var


def visualize_predictions(
    model,
    model_name,
    test_dataloader,
    num_samples=None,
    time_step=0.04,
    device="cpu" if not torch.cuda.is_available() else "cuda",
    scene="NBA",
    pred_len=25,
    fine_tuned=False,
    pretrained=False,
    hist_len=8,
):
    file_suffix = "_finetuned" if fine_tuned else "_pretrained" if pretrained else ""
    print(f"start benchmarking {model_name} on {scene} scene")
    # Redirect output to a file
    set_output_file(model_name, scene, file_suffix, hist_len)
    model = model.to(device)
    model.eval()

    FDE_list = []
    ADE_list = []
    NL_ADE_list = []
    MSE_list = []
    MAE_list = []
    FRE_list = []
    ARE_list = []
    error_list = []
    angular_error_list = []

    for i, sample in enumerate(test_dataloader):
        if num_samples is not None and i == num_samples:
            break

        with torch.no_grad():
            batch = [s.unsqueeze(0).to(device) for s in sample]
            FDE, ADE, NL_ADE, MSE, MAE, FRE, ARE, loss_list, angular_error, _, _, _ = (
                model.test_step(batch, -1)
            )

        FDE_list.append(FDE)
        ADE_list.append(ADE)
        NL_ADE_list.append(NL_ADE)
        MSE_list.append(MSE)
        MAE_list.append(MAE)
        FRE_list.append(FRE)
        ARE_list.append(ARE)
        error_list.append(loss_list)
        angular_error_list.append(angular_error)

        percentage = i / len(test_dataloader) * 100
        # Print progress to the console
        print(f"\rProgress: {percentage:.2f}%", end="", file=original_stdout)

        if num_samples is not None:
            percentage = (i / (num_samples - 1)) * 100
            print(f", Progress small: {percentage:.2f}%", end="", file=original_stdout)

    print("\nDone!", file=original_stdout)
    # Concatenate all the lists of arrays in axis 0
    FDE = np.concatenate(FDE_list, axis=0)
    ADE = np.concatenate(ADE_list, axis=0)
    NL_ADE = np.concatenate(NL_ADE_list, axis=0)
    MSE = np.concatenate(MSE_list, axis=0)
    MAE = np.concatenate(MAE_list, axis=0)
    FRE = np.concatenate(FRE_list, axis=0)
    ARE = np.concatenate(ARE_list, axis=0)

    error = np.concatenate(error_list, axis=0)
    error_mean = error.mean(axis=0)
    error_var = error.std(axis=0)

    angular_error = np.concatenate(angular_error_list, axis=0)
    angular_mean = moving_average(angular_error.mean(axis=0))
    angular_var = moving_average(angular_error.std(axis=0))

    # Capture metrics at specific time steps
    time_steps = [24, 49, 74, 99]
    error_at_steps = [
        error_mean[ts] if ts < len(error_mean) else None for ts in time_steps
    ]
    error_var_at_steps = [
        error_var[ts] if ts < len(error_var) else None for ts in time_steps
    ]
    angular_error_at_steps = [
        angular_error[:, ts].mean() if ts < angular_error.shape[1] else None
        for ts in time_steps
    ]
    angular_error_var_at_steps = [
        angular_error[:, ts].std() if ts < angular_error.shape[1] else None
        for ts in time_steps
    ]

    metrics = {
        "Metric": [
            "FDE",
            "ADE",
            "NL_ADE",
            "MSE",
            "MAE",
            "FRE (Final Radian Error)",
            "ARE (Average Radian Error)",
        ],
        "Mean": [
            FDE.mean(),
            ADE.mean(),
            NL_ADE.mean(),
            MSE.mean(),
            MAE.mean(),
            FRE.mean(),
            ARE.mean(),
        ],
        "Std": [
            FDE.std(),
            ADE.std(),
            NL_ADE.std(),
            MSE.std(),
            MAE.std(),
            FRE.std(),
            ARE.std(),
        ],
        **{
            f"Error at step {ts} (mean)": error
            for ts, error in zip(time_steps, error_at_steps)
        },
        **{
            f"Error at step {ts} (var)": error
            for ts, error in zip(time_steps, error_var_at_steps)
        },
        **{
            f"Angular Error at step {ts} (mean)": error
            for ts, error in zip(time_steps, angular_error_at_steps)
        },
        **{
            f"Angular Error at step {ts} (var)": error
            for ts, error in zip(time_steps, angular_error_var_at_steps)
        },
    }

    df = pd.DataFrame(metrics)
    dir = f"../benchmark/{scene}/{hist_len}/{model_name}{file_suffix}"
    os.makedirs(dir, exist_ok=True)
    df.to_csv(f"{dir}/metrics.csv", index=False)

    def print_error_at_time_step(time_step):
        print(f"Error at time step {(time_step + 1) * 0.04}s:")
        print("\u2500" * 30)
        print(f"Distance: {error_mean[time_step]:.3f} ± {error_var[time_step]:.3f}")
        print(
            f"Angle: {angular_error[:, time_step].mean():.3f} ± {angular_error[:, time_step].std():.3f}"
        )
        print("\u2500" * 30)
        print()

    print("start benchmarking", file=original_stdout)

    print_error_at_time_step(24)
    print_error_at_time_step(49)
    print_error_at_time_step(74)
    print_error_at_time_step(99)
    
    print("Metrics: ")
    print("\u2500" * 30)
    print(f"Mean FDE: {FDE.mean():.3f} ± {FDE.std():.3f}")
    print(f"Mean ADE: {ADE.mean():.3f} ± {ADE.std():.3f}")
    print(f"Mean NL_ADE: {NL_ADE.mean():.3f} ± {NL_ADE.std():.3f}")
    print(f"Mean MSE: {MSE.mean():.3f} ± {MSE.std():.3f}")
    print(f"Mean MAE: {MAE.mean():.3f} ± {MAE.std():.3f}")
    print(f"Mean FRE (Final Radian Error): {FRE.mean():.3f} ± {FRE.std():.3f}")
    print(f"Mean ARE (Average Radian Error): {ARE.mean():.3f} ± {ARE.std():.3f}")
    print("\u2500" * 30)

    time_axis = np.arange(0, error_mean.shape[0]) * time_step

    print("start plotting", file=original_stdout)

    plot(
        time_axis,
        error_mean,
        error_var,
        angular_mean,
        angular_var,
        model_name,
        scene,
        file_suffix,
        hist_len=hist_len,
    )

    # Reset output to console
    reset_output()

    return error_mean, error_var
