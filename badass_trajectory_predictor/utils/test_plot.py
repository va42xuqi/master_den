import os

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_error_angular_error(distances, angles, scene, model_name):
    plt.figure(figsize=(10, 10))
    plt.plot(angles, distances, "o")
    plt.title("Error of the model")
    plt.xlabel("Angular distance")
    plt.ylabel("Error")
    plt.savefig(f"../plots/{scene}/{model_name}_angular_distance_error.png")

    return distances, angles


def plot(
    time_axis, error_mean, error_var, angular_mean, angular_var, model_name, scene
):
    plt.figure(figsize=(10, 10))
    plt.errorbar(time_axis, error_mean, yerr=error_var, fmt="o")
    plt.title("Error of the model")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.savefig(f"../plots/{scene}/{model_name}_error.png")

    plt.figure(figsize=(10, 10))
    plt.errorbar(time_axis, angular_mean, yerr=angular_var, fmt="o")
    plt.title("Angular Error of the model")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.savefig(f"../plots/{scene}/{model_name}_angular_error.png")

    # save mean and variance of angular error to a file
    os.makedirs(f"../plots/{scene}/values", exist_ok=True)
    np.save(f"../plots/{scene}/values/{model_name}_error_mean.npy", error_mean)
    np.save(f"../plots/{scene}/values/{model_name}_error_var.npy", error_var)
    np.save(f"../plots/{scene}/values/{model_name}_angular_mean.npy", angular_mean)
    np.save(f"../plots/{scene}/values/{model_name}_angular_var.npy", angular_var)
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
):
    if model_name != "no_model":
        checkpoint_name = f"{model_name}.ckpt"
        checkpoint = torch.load(
            f"checkpoints/{scene.lower()}/{checkpoint_name}",
            map_location=torch.device(device),
        )
        model.load_state_dict(checkpoint["state_dict"])
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
        print(f"\r Progress: {percentage:.2f}%", end="")
        if num_samples is not None:
            percentage = (i / (num_samples - 1)) * 100
            print(f", Progress small: {percentage:.2f}%", end="")
        torch.cuda.empty_cache()

    print("\nDone!")
    # concatinate all the lists of arrays in axis 0
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
    angular_mean = angular_error.mean(axis=0)
    angular_var = angular_error.std(axis=0)

    print("\u2500" * 30)
    print("\n Metrics: ")
    print(f"Mean FDE: {FDE.mean():.3f} ± {FDE.std():.3f}")
    print(f"Mean ADE: {ADE.mean():.3f} ± {ADE.std():.3f}")
    print(f"Mean NL_ADE: {NL_ADE.mean():.3f} ± {NL_ADE.std():.3f}")
    print(f"Mean MSE: {MSE.mean():.3f} ± {MSE.std():.3f}")
    print(f"Mean MAE: {MAE.mean():.3f} ± {MAE.std():.3f}")
    print(f"Mean FRE (Final Radian Error): {FRE.mean():.3f} ± {FRE.std():.3f}")
    print(f"Mean ARE (Average Radian Error): {ARE.mean():.3f} ± {ARE.std():.3f}")
    print("\u2500" * 30)

    def print_error_at_time_step(time_step):
        print("\u2500" * 30)
        print(f"Error at time step {(time_step + 1) * 0.04}s:")
        print(f"Distance: {error_mean[time_step]:.3f} ± {error_var[time_step]:.3f}")
        print(
            f"Angle: {angular_error[:, time_step].mean():.3f} ± {angular_error[:, time_step].std():.3f}"
        )
        print("\u2500" * 30)

    # TODO: round 2 decimal places
    print_error_at_time_step(24)
    print_error_at_time_step(49)
    print_error_at_time_step(74)
    print_error_at_time_step(99)

    time_axis = np.arange(0, error_mean.shape[0]) * time_step

    plot(time_axis, error_mean, error_var, angular_mean, angular_var, model_name, scene)

    return error_mean, error_var


# load the error_means over all models and combine them in a plot
def plot_all(scene):
    # get the directory of badass_trajectory_predictor (it's not the current directory)
    import badass_trajectory_predictor

    root_path = os.path.dirname(badass_trajectory_predictor.__file__)
    path = os.path.join(root_path, "..", "plots", scene, "values")
    error_means = []
    error_vars = []
    model_names = []

    angular_means = []
    angular_vars = []
    model_names_angular = []  # will be merged with model_names later

    for file in os.listdir(path):
        if "error_mean" in file:
            error_means.append(np.load(os.path.join(path, file)))
            model_names.append(file[:-15])
        if "error_var" in file:
            error_vars.append(np.load(os.path.join(path, file)))
        if "angular_mean" in file:
            angular_means.append(np.load(os.path.join(path, file)))
            model_names_angular.append(file[:-17])
        if "angular_var" in file:
            angular_vars.append(np.load(os.path.join(path, file)))
    # chose a random but significant different color for each model
    colors = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
    ]

    time_axis = np.arange(0, error_means[0].shape[0]) * 0.04

    plt.figure(figsize=(10, 10))
    for i in range(len(error_means)):
        plt.plot(time_axis, error_means[i], label=model_names[i], color=colors[i])
        # plot the variance as a shaded area around the mean
        if fill := False:
            plt.fill_between(
                time_axis,
                error_means[i] - error_vars[i],
                error_means[i] + error_vars[i],
                color=colors[i],
                alpha=0.1,
            )
    plt.xlabel("Time step [s]")
    plt.ylabel("Distance [m]")
    plt.title("Error of the models")
    plt.legend()
    path = os.path.join(root_path, "..", "plots", scene)
    plt.savefig(os.path.join(path, "all_models_error.png"))

    plt.figure(figsize=(10, 10))
    for i in range(len(angular_means)):
        plt.plot(time_axis, angular_means[i], label=model_names[i], color=colors[i])
        # plot the variance as a shaded area around the mean
        if fill := False:
            plt.fill_between(
                time_axis,
                angular_means[i] - angular_vars[i],
                angular_means[i] + angular_vars[i],
                color=colors[i],
                alpha=0.1,
            )
    plt.xlabel("Time step [s]")
    plt.ylabel("Angle [°]")
    plt.title("Angular Error of the models")
    plt.legend()
    path = os.path.join(root_path, "..", "plots", scene)
    plt.savefig(os.path.join(path, "all_models_angular_error.png"))


if __name__ == "__main__":
    plot_all("SOC")
