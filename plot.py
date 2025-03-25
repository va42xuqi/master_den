import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

nba = 'NBA'
soccer = 'SOC'

input_lengths = [1, 10, 25, 50, 100]
excluded_models = ['1l', '2l', 'linear', "trafo", "tf", "bitnet", "lstm"]
excluded_modes = ["pretrained", "other_team", "finetuned", "uni", "pos", "vel"]

def read_results(scene, input_length, models):
    # Define the folder path
    folder = f'benchmark/{scene}/{input_length}/'

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

            # Exclude specified models and modes
            if model not in subfolder:
                continue
            if any(x in subfolder for x in excluded_modes):
                continue

            # Ensure it's a directory
            if os.path.isdir(subfolder_path):
                # Initialize list to collect mean errors for all timesteps
                mean_errors = []
                angular_error = []

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
                    if file == "angular_mean.npy":
                        file_path = os.path.join(subfolder_path, file)
                        try:
                            # Load the .npy file (assuming it's a 1D array)
                            angular_error = np.load(file_path)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

                # Append to results if mean errors were found
                if mean_errors:
                    mean_errors = np.mean(mean_errors, axis=0)  # Average across all loaded files

                    # Slice the array to include only the first second (25 timesteps)
                    mean_errors = mean_errors[:50]
                    angular_error = angular_error[:50]

                    results.append((subfolder, mean_errors, angular_error))
        
    return results

# Placeholder function to change the subfolder name
def change_name_style(name):
    # Leave this function blank as requested
    return name

def plot_results(results, scene, input_length, models):
    # Create a DataFrame for easier plotting with seaborn
    data = []
    max_errors_at_1s = {}

    print(f"Results for scene: {scene}, input length: {input_length}\n")
    
    for subfolder, mean_errors, angular_error in results:
        label = change_name_style(subfolder)

        # Convert angular error to degrees
        angular_error_degrees = np.degrees(angular_error)

        # Calculate metrics
        final_displacement_error = mean_errors[-1]  # FDE as last timestep error
        average_displacement_error = np.mean(mean_errors)  # ADE as mean of errors

        # Assuming angular_error represents difference from origin
        final_angular_error = angular_error_degrees[-1]  # Final angular error in degrees
        average_angular_error = np.mean(angular_error_degrees)  # Average angular error

        # Print metrics for the current model
        print(f"Model: {label}")
        print(f"  Final Displacement Error (FDE): {final_displacement_error:.4f}")
        print(f"  Average Displacement Error (ADE): {average_displacement_error:.4f}")
        print(f"  Final Angular Error (FAE): {final_angular_error:.4f}°")
        print(f"  Average Angular Error (AAE): {average_angular_error:.4f}°\n")

        # Shift the time axis to start from 0.04 seconds and append 0 error at time 0
        timesteps = np.arange(1, len(mean_errors) + 1) * 0.04  # Time from 0.04 to 1s
        timesteps = np.insert(timesteps, 0, 0)  # Insert 0 at the beginning
        mean_errors = np.insert(mean_errors, 0, 0)  # Insert 0 error at the beginning

        data.extend([(label, timestep, error) for timestep, error in zip(timesteps, mean_errors)])

        # Capture the maximum error at 1 second (last point)
        max_error_at_1s = mean_errors[-1]
        max_errors_at_1s[label] = max_error_at_1s

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['Model', 'Time (s)', 'Mean Error'])

    # Plot using Seaborn
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Time (s)', y='Mean Error', hue='Model', marker='o')

    # Add horizontal lines showing the max error at 1 second for each model
    for label, max_error in max_errors_at_1s.items():
        # Use a lighter color and thinner line style for the horizontal line
        plt.axhline(y=max_error, linestyle='--', color='gray', alpha=0.5)

        # Annotate the max error at 1s next to the line
        plt.text(1.06, max_error, f'{max_error:.2f}', color='black', va='center', ha='left', fontsize=10)

    # Customize plot
    plt.title(f'Model Performance Over Time - {scene} (Input Length {input_length})')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Error')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Ensure the 'defense/plot' directory exists
    os.makedirs('defense/plot/compare', exist_ok=True)

    # Save the plot to the specified directory
    output_filename = f'defense/plot/compare/results_{scene}_{input_length}.png'
    plt.savefig(output_filename)
    print(f'Plot saved as {output_filename}')



all_models = ['lstm', 'lmu', 'trafo', 'bitnet']
# Example usage
scene = nba  # e.g., 'NBA'
input_length = 50

results = read_results(scene, input_length, all_models)
plot_results(results, scene, input_length, all_models)
