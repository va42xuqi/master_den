import os
import pandas as pd

# Directory containing the CSV files
directory = "results/tables"
# Directories to save the CSV and LaTeX files
csv_output_directory_pos = "results/pos_vel/pos"
csv_output_directory_vel = "results/pos_vel/vel"
latex_output_directory_pos = "tables/pos_vel/pos"
latex_output_directory_vel = "tables/pos_vel/vel"
# Ensure the output directories exist
os.makedirs(csv_output_directory_pos, exist_ok=True)
os.makedirs(csv_output_directory_vel, exist_ok=True)
os.makedirs(latex_output_directory_pos, exist_ok=True)
os.makedirs(latex_output_directory_vel, exist_ok=True)

# Function to process files for a specific scene type and keyword
def process_files(scene_type, keyword, csv_output_directory, latex_output_directory):
    # Collect all relevant CSV files for the given scene type and keyword
    relevant_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv") and scene_type in filename and keyword in filename and "50" in filename:
                if not any(
                    keyword in filename
                    for keyword in ["finetuned", "pretrained"]
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

    metrics_to_remove = [
    'Rad Err 1s', 'Rad Err 2s', 'Rad Err 3s',
    'Err 1s', 'Err 2s', 'Err 3s', 'Err 4s'
]

    combined_df = combined_df[~combined_df['Metric'].isin(metrics_to_remove)]

    rename_dict = {
        'Rad Err 4s': 'Rad Err (grad)',
        'ADE': "ADE (m)",
        "FDE": "FDE (m)",
        "ARE": "ARE (grad)",
        "FRE": "FRE (grad)",
        "MAE": "MAE (m)",
        "MSE": "MSE (m)",
        "NL\_ADE": "NL\_ADE (m)",
        }
    combined_df.index = combined_df.index.to_series().replace(rename_dict)
    combined_df['Metric'] = combined_df['Metric'].replace(rename_dict) 

    # Pivot the DataFrame to have models as columns
    pivot_df = combined_df.pivot(
        index="Metric", columns="Model", values="Mean\_Std"
    ).reset_index()

    combined_df = combined_df.reindex((["MAE (m)", "MSE (m)", "FDE (m)", "ADE (m)", "FRE (grad)", "NL\_ADE (m)" "ARE (grad)", "Rad Err (grad)"]), axis = "rows")

    # Split the pivoted DataFrame into two parts
    mid_point = len(pivot_df.columns) // 2 + 1
    first_part = pivot_df.iloc[:, :mid_point]
    second_part = pivot_df.iloc[:, [0] + list(range(mid_point, len(pivot_df.columns)))]

    # Save each part to a separate CSV file
    csv_file_path_1 = os.path.join(
        csv_output_directory, f"pivoted_data_50_{scene_type}_{keyword}_part1.csv"
    )
    first_part.to_csv(csv_file_path_1, index=False)

    csv_file_path_2 = os.path.join(
        csv_output_directory, f"pivoted_data_50_{scene_type}_{keyword}_part2.csv"
    )
    second_part.to_csv(csv_file_path_2, index=False)

    print(f"CSV files for {scene_type} ({keyword}) saved to {csv_file_path_1} and {csv_file_path_2}")

    # Convert each part to LaTeX format with longtable
    latex_table_1 = first_part.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l" + "c" * (first_part.shape[1] - 1),
        caption=f"Results for {scene_type} scene type ({keyword} Part 1).",
        label=f"{scene_type}_{keyword}_results_part1",
        escape=False  # To handle special characters if needed
    )
    latex_table_2 = second_part.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l" + "c" * (second_part.shape[1] - 1),
        caption=f"Results for {scene_type} scene type ({keyword} Part 2).",
        label=f"{scene_type}_{keyword}_results_part2",
        escape=False  # To handle special characters if needed
    )

    latex_table_1 = latex_table_1.replace("\\begin{table}", "\\begin{table}[H]\n\\centering")
    latex_table_2 = latex_table_2.replace("\\begin{table}", "\\begin{table}[H]\n\\centering")

    # Save each LaTeX table to a separate .tex file
    latex_file_path_1 = os.path.join(
        latex_output_directory, f"pivoted_data_50_{scene_type}_{keyword}_part1.tex"
    )
    with open(latex_file_path_1, "w") as f:
        f.write(latex_table_1)

    latex_file_path_2 = os.path.join(
        latex_output_directory, f"pivoted_data_50_{scene_type}_{keyword}_part2.tex"
    )
    with open(latex_file_path_2, "w") as f:
        f.write(latex_table_2)

    print(f"LaTeX tables for {scene_type} ({keyword}) saved to {latex_file_path_1} and {latex_file_path_2}")


# Process SOC and NBA scene types for 'pos' and 'vel'
process_files("SOC", "pos", csv_output_directory_pos, latex_output_directory_pos)
process_files("SOC", "vel", csv_output_directory_vel, latex_output_directory_vel)
process_files("NBA", "pos", csv_output_directory_pos, latex_output_directory_pos)
process_files("NBA", "vel", csv_output_directory_vel, latex_output_directory_vel)
