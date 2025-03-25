import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

nba = 'NBA'
soccer = 'SOC'

input_lengths = [1, 10, 25, 50, 100]
excluded_models = ['1l', '2l', 'linear', "trafo", "tf", "bitnet", "lstm"]
excluded_modes = ["other_team", "finetuned", "uni", "pos", "vel"]

def read_results(scene, input_length, model):
    """
    Reads subfolders for the single 'model' within benchmark/<scene>/<input_length>/.
    Returns a list of (subfolder, mean_errors).
    """
    folder = f'benchmark/{scene}/{input_length}/'
    results = []

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        # Must match model name and exclude certain modes
        if model not in subfolder:
            continue
        if any(x in subfolder for x in excluded_modes):
            continue

        if os.path.isdir(subfolder_path):
            mean_errors_list = []
            for file in os.listdir(subfolder_path):
                if file == "error_mean.npy":
                    file_path = os.path.join(subfolder_path, file)
                    try:
                        arr = np.load(file_path)
                        mean_errors_list.append(arr)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            if mean_errors_list:
                mean_errors = np.mean(mean_errors_list, axis=0)
                mean_errors = mean_errors[:25]  # up to 1s (25 timesteps)
                results.append((subfolder, mean_errors))

    return results

def plot_improvement_single_vs_all(results, scene, input_length, model):
    """
    1) Find the single-object (subfolder containing 'pretrained') and 
       the all-objects (subfolder not containing 'pretrained').
    2) Compute improvement: 100 * (single - all) / single.
    3) Produce ONE figure for this specific model.
    """
    # Identify single vs. all
    single_err = None
    all_err = None

    single_subf = None
    all_subf = None

    for (subfolder, mean_errors) in results:
        if "pretrained" in subfolder.lower():
            single_err = mean_errors
            single_subf = subfolder
        else:
            all_err = mean_errors
            all_subf = subfolder

    # If we didn't find both, just skip plotting
    if single_err is None or all_err is None:
        print(f"No valid single/all pair found for model: {model}")
        return

    # Compute improvement
    improvements = []
    for s_err, a_err in zip(single_err, all_err):
        if s_err == 0:
            improvements.append(0.0)  # or handle differently
        else:
            improvements.append(100.0 * (s_err - a_err) / s_err)

    timesteps = np.arange(1, len(single_err) + 1) * 0.04  # 0.04..1.0 s

    df_plot = pd.DataFrame({
        'Time (s)': timesteps,
        'Improvement (%)': improvements
    })

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_plot, 
        x='Time (s)', 
        y='Improvement (%)',
        marker='o', 
        label=f"{all_subf} vs {single_subf}"
    )

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f"Improvement: All vs Single - {model} (Input={input_length}, Scene={scene})")
    plt.xlabel("Time (s)")
    plt.ylabel("Improvement (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save
    os.makedirs('defense/plot/compare', exist_ok=True)
    out_fn = f"defense/plot/compare/improvement_all_vs_single_{scene}_{model}.png"
    plt.savefig(out_fn)
    print(f"Plot saved as {out_fn}")

# ----------------------------------------
# Main script to run for all models
# ----------------------------------------
if __name__ == "__main__":
    scene = nba
    input_length = 50
    all_models = ['lstm', 'lmu', 'trafo', 'bitnet']

    for mdl in all_models:
        # 1) Read results for this model
        results_for_mdl = read_results(scene, input_length, mdl)
        # 2) Plot improvement for single vs. all
        plot_improvement_single_vs_all(results_for_mdl, scene, input_length, mdl)
