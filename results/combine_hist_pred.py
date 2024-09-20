import os
import pandas as pd
import re
import numpy as np


def remove_std(value):
    return value.split()[0]


# Define a function to convert rad to deg
def convert_rad_to_deg(value):
    # Extract mean and std using regex
    match = re.match(r"([\d.]+) \\pm ([\d.]+)", value)
    if match:
        mean_rad = float(match.group(1))
        std_rad = float(match.group(2))
        # Convert radians to degrees
        mean_deg = mean_rad * 180 / np.pi
        std_deg = std_rad * 180 / np.pi

        # Reformat the string with the converted values
        return f"{mean_deg:.2f} \\pm {std_deg:.2f}"
    else:
        return value

# Directory containing the CSV files
directory = "results/tables"
# Directory to save the CSV files
csv_output_directory = "results/hist_pred"
# Directory to save the LaTeX files
latex_output_directory = "tables/hist_pred"
# Ensure the output directories exist
os.makedirs(csv_output_directory, exist_ok=True)
os.makedirs(latex_output_directory, exist_ok=True)


# Function to process files for a specific scene type
def process_files(scene_type, i):
    # Collect all relevant CSV files for the given scene type
    relevant_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):
                if f"_{i}_" in filename and scene_type in filename:
                    if not any(
                        keyword in filename
                        for keyword in ["pos", "vel", "uni", "finetuned", "pretrained"]
                    ):
                        relevant_files.append(os.path.join(root, filename))

    # Initialize an empty list to store DataFrames
    data_frames = []

    # Read and process each CSV file
    for file_path in relevant_files:
        df = pd.read_csv(file_path)
        # Extract model name from filename
        model_name = os.path.basename(file_path).replace(".csv", "")
        if "lmu" in model_name:
            model_name = "LMU"
        if "lstm" in model_name:
            model_name = "LSTM"
        if "1l_linear" in model_name:
            model_name = "one layer linear"
        if "2l_linear" in model_name:
            model_name = "two layer linear"
        if "tf" in model_name or "trafo" in model_name:
            model_name = "Transformer"
        if "bitnet" in model_name:
            model_name = "BitNet"
        df["Model"] = model_name
        # Append DataFrame to the list
        data_frames.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(data_frames, ignore_index=True)

    
    # Rename columns and indices
    rename_dict = {
        'ADE': "ADE (m)",
        "FDE": "FDE (m)",
        "ARE": "ARE (grad)",
        "FRE": "FRE (grad)",
        "MAE": "MAE (m)",
        "MSE": "MSE (m)",
        "NL_ADE": "NL_ADE (m)",
        "Err 1s": "Disp. Err. 1s (m)",
        "Err 2s": "Disp. Err. 2s (m)",
        "Err 3s": "Disp. Err. 3s (m)",
        "Err 4s": "Disp. Err. 4s (m)",
        "Rad Err 1s": "Rad. Err. 1s (grad)",
        "Rad Err 2s": "Rad. Err. 2s (grad)",
        "Rad Err 3s": "Rad. Err. 3s (grad)",
        "Rad Err 4s": "Rad. Err. 4s (grad)",

    }
    combined_df['Metric'] = combined_df['Metric'].replace(rename_dict)

    metrics_to_remove = [
        "FDE (m)", "FRE (grad)"
    ]

    # Filter out rows based on the metrics to be removed
    combined_df = combined_df[~combined_df['Metric'].isin(metrics_to_remove)]

    # Pivot the DataFrame to have models as columns
    pivot_df = combined_df.pivot(
        index="Metric", columns="Model", values="Mean\_Std"
    ).reset_index()

    rad_metrics = ["Rad. Err. 1s (grad)", "Rad. Err. 2s (grad)", "Rad. Err. 3s (grad)", "Rad. Err. 4s (grad)"]
    # Apply the conversion only to relevant metrics
    for metric in rad_metrics:
        if metric in pivot_df['Metric'].values:
            for col in pivot_df.columns[1:]:  # Skip the 'Metric' column
                pivot_df.loc[pivot_df['Metric'] == metric, col] = pivot_df.loc[pivot_df['Metric'] == metric, col].apply(convert_rad_to_deg)
        
    for metric in pivot_df["Metric"]:
        for col in pivot_df.columns[1:]:  # Skip the 'Metric' column
            pivot_df.loc[pivot_df['Metric'] == metric, col] = pivot_df.loc[pivot_df['Metric'] == metric, col].apply(remove_std)

    for key in pivot_df.keys():
        if i in key:
            del pivot_df[key]


    # Save each part to a separate CSV file
    csv_file_path_1 = os.path.join(
        csv_output_directory, f"pivoted_data_{i}_{scene_type}_full.csv"
    )
    pivot_df.to_csv(csv_file_path_1, index=False)

    print(f"CSV files for {scene_type} saved to {csv_file_path_1}")

    # Convert each part to LaTeX format with longtable
    latex_table_1 = pivot_df.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l" + "c" * 4,
        caption=f"Results for {scene_type} on history length of {int(i)*0.04}s.",
        label=f"hist:{scene_type}_{int(i)*0.04}s",
        escape=False  # To handle special characters if needed
    )

    latex_table_1 = latex_table_1.replace("\\begin{table}", "\\begin{table}[H]\n\\centering")

    # Save each LaTeX table to a separate .tex file
    latex_file_path_1 = os.path.join(
        latex_output_directory, f"pivoted_data_{i}_{scene_type}_full.tex"
    )
    with open(latex_file_path_1, "w") as f:
        f.write(latex_table_1)

    print(f"LaTeX tables for {scene_type} saved to {latex_file_path_1}")


# Process both SOC and NBA scene types
for i in ["1", "25", "50"]:
    process_files("SOC", i)
    process_files("NBA", i)
