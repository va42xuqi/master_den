import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

nba = 'NBA'
soccer = 'SOC'

# 0.4s corresponds to input_length = 10
input_lengths = [10, 25, 50, 100]
excluded_models = ['1l', '2l', 'linear', "trafo", "tf", "bitnet", "lstm"]
excluded_modes = ["pretrained", "other_team", "finetuned", "uni", "vel", "pos"]

def read_results(scene, input_length, model):
    folder = f'benchmark/{scene}/{input_length}/'
    results = []

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        # Skip if model name isn't in subfolder, or if it contains excluded modes
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
                # Average over all files found
                mean_errors = np.mean(mean_errors, axis=0)
                # Only keep first 25 timesteps (up to 1 second if 0.04s increments)
                mean_errors = mean_errors[:25]
                results.append((subfolder, mean_errors))

    return results

def change_name_style(name):
    # Adjust legend labels if desired
    return name

def plot_results(scene, input_lengths, model, baseline_input_length=10):
    """
    Plots each curve's improvement vs. the baseline (input_length=10 => 0.4s).
    time-step 1 => 0.04s, time-step 2 => 0.08s, etc., with no time=0 inserted.
    """
    # 1) Collect all data in a single DataFrame
    data = []
    for inp_len in input_lengths:
        results_for_len = read_results(scene, inp_len, model)
        for subfolder, mean_errors in results_for_len:
            label = change_name_style(subfolder)
            timesteps = np.arange(1, len(mean_errors) + 1) * 0.04
            for t, err in zip(timesteps, mean_errors):
                data.append({
                    'Model':      label,
                    'Time (s)':   t,
                    'Mean Error': err,
                    'InputLen':   inp_len
                })

    df = pd.DataFrame(data)

    # 2) Identify baseline (input_len=10 => 0.4s)
    df_baseline = df[df['InputLen'] == baseline_input_length].copy()
    df_baseline.rename(columns={'Mean Error': 'Baseline Error'}, inplace=True)
    df_baseline = df_baseline[['Time (s)', 'Baseline Error']]

    # 3) Merge and compute improvement for other input lengths
    dfs_improve = []
    for inp_len in input_lengths:
        if inp_len == baseline_input_length:
            # We skip plotting the baseline as an "improvement" curve
            continue
        
        df_curr = df[df['InputLen'] == inp_len].copy()
        # Match on Time (s)
        df_merged = pd.merge(df_curr, df_baseline, on='Time (s)', how='inner')

        # Improvement = 100 * (baseline - new) / baseline
        df_merged['Error Improvement (%)'] = np.where(
            df_merged['Baseline Error'] == 0,
            0.0,
            (df_merged['Baseline Error'] - df_merged['Mean Error']) 
            / df_merged['Baseline Error'] * 100
        )

        # Label curve
        df_merged['Curve Label'] = f"{inp_len * 0.04:.1f}s vs 0.4s"
        dfs_improve.append(df_merged)

    if not dfs_improve:
        print("No improvement data to plot.")
        return

    df_improvement_all = pd.concat(dfs_improve, ignore_index=True)

    # 4) Plot the improvement
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_improvement_all,
        x='Time (s)',
        y='Error Improvement (%)',
        hue='Curve Label',
        style='Model',
        markers=True
    )

    plt.title(f'Percentage Improvement over Baseline (0.4s) - {model}')
    plt.xlabel('Time (s)')
    plt.ylabel('Error Improvement (%)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True)
    plt.tight_layout()

    # 5) Save the plot
    out_dir = 'defense/plot/less'
    os.makedirs(out_dir, exist_ok=True)
    output_filename = os.path.join(out_dir, f'percentage_improvement_from_baseline_{scene}_{model}.png')
    plt.savefig(output_filename)
    print(f'Plot saved as {output_filename}')

# -------------------------------------
# Example usage
# -------------------------------------
all_models = ['lstm', 'lmu', 'trafo', 'bitnet']
scene = nba

for mdl in all_models:
    plot_results(scene, input_lengths, mdl, baseline_input_length=10)
