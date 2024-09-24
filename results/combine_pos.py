import os
import pandas as pd
import numpy as np
 
# Directory containing the CSV files
directory = "results/tables"
# Directory to save the CSV files
csv_output_directory = "results/pos"
# Directory to save the LaTeX files
latex_output_directory = "tables/pos"
# Ensure the output directories exist
os.makedirs(csv_output_directory, exist_ok=True)
os.makedirs(latex_output_directory, exist_ok=True)
 
 
# Function to process files for a specific scene type
def process_files(scene_type):
    # Collect all relevant CSV files for the given scene type
    relevant_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):
                if "50" in filename and scene_type in filename and "pos" in filename:
                    if not any(
                        keyword in filename
                        for keyword in ["vel", "uni", "finetuned", "pretrained", "other"]
                    ):
                        relevant_files.append(os.path.join(root, filename))
 
    # Initialize an empty list to store DataFrames
    data_frames = []
 
    # Read and process each CSV file
    for file_path in relevant_files:
        df = pd.read_csv(file_path)
        # Extract model name from filename
        model_name = os.path.basename(file_path).replace(".csv", "")
        model_name = model_name.split("50_")[1]
        model_name = model_name.replace("os", "")
        model_name = model_name.replace("_", " ")
        if "tf" in model_name or "trafo" in model_name:
            model_name = "Trafo"
        if "bitnet" in model_name:
            model_name = "BitNet"
        if "1l" in model_name or "one_layer" in model_name:
            model_name = "1L Linear"
        if "2l" in model_name or "two_layer" in model_name:
            model_name = "2L Linear"
        if "lmu" in model_name:
            model_name = "LMU"
        if "lstm" in model_name:
            model_name = "LSTM"
        df["Model"] = model_name
 
        # Replace '\\pm' with '±' in the "Mean_Std" column
        df['Mean_Std'] = df['Mean\_Std'].str.replace(r'\\pm', '±', regex=True)
 
        # Split the "Mean_Std" column into two separate columns
        df[['Mean', 'Std']] = df['Mean_Std'].str.split('±', expand=True).astype(float)
        # Convert AAE and FAE from radians to degrees
        if 'ARE' in df['Metric'].values or 'FRE' in df['Metric'].values:
            df.loc[df['Metric'].str.contains('ARE|FRE'), 'Mean'] *= (180 / np.pi)
            df.loc[df['Metric'].str.contains('ARE|FRE'), 'Std'] *= (180 / np.pi)
 
        # Append DataFrame to the list
        data_frames.append(df)
        # Rename columns and indices
 
    # Concatenate all DataFrames
    combined_df = pd.concat(data_frames, ignore_index=True)
    rename_dict = {
        'ADE': "ADE (\\si{\\meter})",
        "FDE": "FDE (\\si{\\meter})",
        "ARE": "AAE (\\si{\\text{grad}})",
        "FRE": "FAE (\\si{\\text{grad}})",
        "MAE": "MAE (\\si{\\meter})",
        "MSE": "MSE (\\si{\\meter})",
        "NL\\_ADE": "NL\\_ADE (\\si{\\meter})",
        "Err 1s": "Disp. Err. 1s (\\si{\\meter})",
        "Err 2s": "Disp. Err. 2s (\\si{\\meter})",
        "Err 3s": "Disp. Err. 3s (\\si{\\meter})",
        "Err 4s": "Disp. Err. 4s (\\si{\\meter})",
        "Rad Err 1s": "Rad. Err. 1s (\\si{\\text{grad}})",
        "Rad Err 2s": "Rad. Err. 2s (\\si{\\text{grad}})",
        "Rad Err 3s": "Rad. Err. 3s (\\si{\\text{grad}})",
        "Rad Err 4s": "Rad. Err. 4s (\\si{\\text{grad}})",
    }
    metrics_to_keep = [
    "ADE (\\si{\\meter})",
    "AAE (\\si{\\text{grad}})",
    "FDE (\\si{\\meter})",
    "FAE (\\si{\\text{grad}})",
    "MAE (\\si{\\meter})",
    "MSE (\\si{\\meter})",
    "NL\\_ADE (\\si{\\meter})",
    ]
    combined_df['Metric'] = combined_df['Metric'].replace(rename_dict)
    combined_df = combined_df[combined_df['Metric'].isin(metrics_to_keep)]
 
 
    # Create a DataFrame with 'Mean' and 'Std' stacked as separate rows
    mean_df = combined_df.pivot(index='Metric', columns='Model', values='Mean')
    mean_df['Statistic'] = 'Mean'
 
    std_df = combined_df.pivot(index='Metric', columns='Model', values='Std')
    std_df['Statistic'] = 'Std'
 
    # Combine mean and std into a single DataFrame
    combined_pivot_df = pd.concat([mean_df, std_df]).reset_index()
 
    # Sort the DataFrame to ensure 'Mean' comes first for each metric
    combined_pivot_df['Statistic'] = pd.Categorical(combined_pivot_df['Statistic'], categories=['Mean', 'Std'], ordered=True)
    combined_pivot_df = combined_pivot_df.sort_values(by=['Metric', 'Statistic']).reset_index(drop=True)
 
    # Set the metric name only for the first occurrence and empty for others
    combined_pivot_df['Metric'] = combined_pivot_df['Metric'].where(combined_pivot_df['Statistic'] == 'Mean')
 
    # Fill NaN values with an empty string
    combined_pivot_df['Metric'] = combined_pivot_df['Metric'].fillna('')
 
    # Reorder the columns to ensure 'Statistic' is the second column
    column_order = ['Metric', 'Statistic'] + [col for col in combined_pivot_df.columns if col not in ['Metric', 'Statistic']]
    combined_pivot_df = combined_pivot_df[column_order]
 
    # Bold the smallest value in each 'Mean' row
    for metric in combined_pivot_df['Metric'].unique():
        metric_mask = combined_pivot_df['Metric'] == metric
        mean_row = combined_pivot_df.loc[metric_mask & (combined_pivot_df['Statistic'] == 'Mean')]
        if not mean_row.empty:
            min_value = mean_row.iloc[:, 2:].min(axis=1).values[0]
            min_columns = mean_row.columns[2:][mean_row.iloc[:, 2:].values[0] == min_value]
            for col in min_columns:
                combined_pivot_df.loc[metric_mask & (combined_pivot_df['Statistic'] == 'Mean'), col] = (
                    "\\textbf{" + combined_pivot_df.loc[metric_mask & (combined_pivot_df['Statistic'] == 'Mean'), col].apply(lambda x: f"{x:.2f}") + "}"
                )
 
    # Save to a single CSV file
    csv_file_path = os.path.join(csv_output_directory, f"pivoted_data_50_{scene_type}.csv")
    combined_pivot_df.to_csv(csv_file_path, index=False)
 
    print(f"CSV file for {scene_type} saved to {csv_file_path}")
 
    # Convert to LaTeX format with longtable
    latex_table = combined_pivot_df.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l|l|" + "|c" * (combined_pivot_df.shape[1] - 2),
        caption=f"Results for {scene_type} scene type.",
        label=f"{scene_type} results",
        escape=False  # To handle special characters if needed
    )
    latex_table = latex_table.replace("\\toprule", "")
    latex_table = latex_table.replace("\\bottomrule", "")
    latex_table = latex_table.replace("\\midrule", "\\hline\\hline")
 
    # Add horizontal lines only after Std rows
    new_lines = []
    for line in latex_table.splitlines():
        new_lines.append(line)
        if 'Std' in line:
            new_lines.append("\\hline")  # Add midrule after Std
 
    latex_table = "\n".join(new_lines)
    latex_table = latex_table.replace("\\begin{table}", "\\begin{table}[H]\n\\centering")
 
    # Save the LaTeX table to a .tex file
    latex_file_path = os.path.join(latex_output_directory, f"pivoted_data_50_{scene_type}.tex")
    with open(latex_file_path, "w") as f:
        f.write(latex_table)
 
    print(f"LaTeX table for {scene_type} saved to {latex_file_path}")
 
 
# Process both SOC and NBA scene types
process_files("SOC")
process_files("NBA")