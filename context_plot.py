import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

nba = 'NBA'
soccer = 'SOC'

input_lengths = [1, 10, 25, 50, 100]
excluded_models = ['1l', '2l', 'linear', "trafo", "tf", "bitnet", "lstm"]
excluded_modes = ["pretrained", "other_team", "finetuned", "uni"]

def read_results(scene, input_length, model):
    folder = f'benchmark/{scene}/{input_length}/'
    results = []

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        # Must match model name and exclude certain modes if needed
        if model not in subfolder:
            continue
        if any(x in subfolder for x in excluded_modes):
            continue

        if os.path.isdir(subfolder_path):
            mean_errors = []
            for file in os.listdir(subfolder_path):
                if file == "error_mean.npy":
                    file_path = os.path.join(subfolder_path, file)
                    try:
                        arr = np.load(file_path)
                        mean_errors.append(arr)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            if mean_errors:
                mean_errors = np.mean(mean_errors, axis=0)  # average over all found .npy files
                mean_errors = mean_errors[:25]              # first second (25 timesteps)
                results.append((subfolder, mean_errors))

    return results

def change_name_style(name):
    return name  # modify legend label if needed

def plot_results_improvement(results, scene, input_length, model):
    """
    1) Identify the 'pos' baseline subfolder.
    2) Compute improvement (%) of 'vel' and 'full' over 'pos'.
    3) Plot improvement vs. time (0.04..1.0s).
    """
    # Separate subfolders into pos, vel, full
    pos_data = None
    vel_data = None
    full_data = None

    for (subfolder, mean_errors) in results:
        lower_name = subfolder.lower()
        if "pos" in lower_name:
            pos_data = mean_errors
        elif "vel" in lower_name:
            vel_data = (subfolder, mean_errors)
        else:
            # If it has neither 'pos' nor 'vel' but matches the model => full input
            full_data = (subfolder, mean_errors)

    # If no pos baseline found, just return
    if pos_data is None:
        print("No position-only ('pos') subfolder found as baseline.")
        return

    # Prepare data for plotting
    data_for_plot = []
    timesteps = np.arange(1, 26) * 0.04  # 1..25 => 0.04..1.0 s

    # Helper: compute improvement(%) = 100 * (pos - other)/pos
    def compute_improvement(pos_err, other_err):
        vals = []
        for p, o in zip(pos_err, other_err):
            if p == 0:
                vals.append(0.0)  # or handle differently if baseline can be zero
            else:
                vals.append(100.0 * (p - o) / p)
        return vals

    # If we have velocity data, compute improvement and store
    if vel_data:
        subf, vel_err = vel_data
        improvements = compute_improvement(pos_data, vel_err)
        for t, imp in zip(timesteps, improvements):
            data_for_plot.append({
                'Variant': change_name_style(subf),
                'Time (s)': t,
                'Improvement (%)': imp
            })

    # If we have full data, compute improvement and store
    if full_data:
        subf, full_err = full_data
        improvements = compute_improvement(pos_data, full_err)
        for t, imp in zip(timesteps, improvements):
            data_for_plot.append({
                'Variant': change_name_style(subf),
                'Time (s)': t,
                'Improvement (%)': imp
            })

    if not data_for_plot:
        print("No 'vel' or 'full' variant found to compare against 'pos'.")
        return

    df_plot = pd.DataFrame(data_for_plot)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_plot,
        x='Time (s)',
        y='Improvement (%)',
        hue='Variant',
        marker='o'
    )

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f"Improvement Over position only Baseline - {scene}, Input={input_length*0.04}s, Model={model}")
    plt.xlabel('Time (s)')
    plt.ylabel('Improvement (%)')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs('defense/plot/context', exist_ok=True)
    output_filename = f'defense/plot/context/improvement_{scene}_{input_length}_{model}.png'
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

# --- Example usage ---
all_models = ['lstm', 'lmu', 'trafo', 'bitnet']
scene = nba
input_length = 50

for mdl in all_models:
    results = read_results(scene, input_length, mdl)
    plot_results_improvement(results, scene, input_length, mdl)
