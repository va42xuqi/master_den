import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

nba = 'NBA'
soccer = 'SOC'

def read_results(scene, input_length, models):
    # Define the folder path
    folder = f'benchmark/plot1/'

    # Initialize an empty list to store tuples (subfolder, mean errors array)
    results = []

    # Check if the folder exists
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []

    for model in models:
        # Traverse through all subfolders in the main directory
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)

            # Ensure it's a directory
            if os.path.isdir(subfolder_path):
                # Initialize list to collect mean errors for all timesteps
                mean_errors = []

                # Traverse files within the subfolder
                for file in os.listdir(subfolder_path):
                    if file == "error_mean.npy":
                        file_path = os.path.join(subfolder_path, file)
                        try:
                            # Load the .npy file (assuming it's a 1D array)
                            mean_error = np.load(file_path)
                            mean_errors.append(mean_error)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

                # Append to results if mean errors were found
                if mean_errors:
                    mean_errors = np.mean(mean_errors, axis=0)  # Average across all loaded files

                    # Slice the array to include only the first 13 timesteps
                    mean_errors = mean_errors[:10]

                    results.append((subfolder, mean_errors))
        
    return results


# Placeholder function to change the subfolder name
def change_name_style(name):
    # Leave this function blank as requested
    return name

def plot_results(results, scene, input_length, models):
    # Create a DataFrame for easier plotting with seaborn
    data = []
    max_errors_at_1s = {}
    average_errors = {}  # Dictionary to store the ADE for each model
    final_errors = {}  # Dictionary to store the FDE for each model

    for subfolder, mean_errors in results:
        label = change_name_style(subfolder)
        
        # Shift the time axis to start from 0.04 seconds and append 0 error at time 0
        timesteps = np.arange(1, len(mean_errors) + 1) * 0.04  # Time from 0.04 to 0.44s (13 points)
        timesteps = np.insert(timesteps, 0, 0)  # Insert 0 at the beginning
        mean_errors = np.insert(mean_errors, 0, 0)  # Insert 0 error at the beginning

        # Add data for seaborn plotting
        data.extend([(label, timestep, error) for timestep, error in zip(timesteps, mean_errors)])

        # Calculate the Average Displacement Error (ADE)
        ade_whole_line = np.mean(mean_errors)
        average_errors[label] = ade_whole_line  # Store for later printing

        # Capture the Final Displacement Error (FDE, last point in the error curve)
        fde_last_point = mean_errors[-1]
        final_errors[label] = fde_last_point  # Store for later printing

        # Capture the maximum error at the last timestep for annotations
        max_errors_at_1s[label] = mean_errors[-1]

    # Print the ADE and FDE for each model
    print("\n--- Model Error Summary ---")
    for label in average_errors.keys():
        print(f"Model: {label}")
        print(f"  ADE (Average Error - whole line): {average_errors[label]:.4f}")
        print(f"  FDE (Final Displacement Error): {final_errors[label]:.4f}")
    print("---------------------------\n")

    # Convert data to DataFrame for plotting
    df = pd.DataFrame(data, columns=['Model', 'Time (s)', 'Mean Error'])

    # Plot using Seaborn
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Time (s)', y='Mean Error', hue='Model', marker='o')

    # Annotate the max error at the last timestep
    for label, max_error in max_errors_at_1s.items():
        plt.text(
            df['Time (s)'].max() + 0.01,  # Position slightly beyond the last timestep
            max_error,
            f'{max_error:.2f}',
            color='black',
            va='center',
            ha='left',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  # Add a background for better visibility
        )

    # Customize plot
    plt.title(f'Model Performance Over Time - {scene} (Input Length {input_length*0.04}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Error')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Set appropriate x and y limits
    plt.xlim(0, df['Time (s)'].max() + 0.05)  # Extend slightly beyond the last point
    plt.ylim(0, None)  # Let matplotlib auto-scale the y-axis above 0

    # Ensure the 'defense/plot' directory exists
    os.makedirs('defense/plot/compare', exist_ok=True)

    # Save the plot to the specified directory
    output_filename = f'defense/plot/compare/results_{scene}_{input_length}.png'
    plt.savefig(output_filename)
    print(f'Plot saved as {output_filename}')


# Example usage
all_models = ['lstm', 'lmu', 'trafo', 'bitnet']
scene = nba  # e.g., 'NBA'
input_length = 50

results = read_results(scene, input_length, all_models)
plot_results(results, scene, input_length, all_models)
