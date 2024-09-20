import os
import pandas as pd
from tqdm import tqdm

# Directory containing the CSV files
directory = "benchmark"
# Directory to save the processed CSV files
csv_output_directory = "results/tables"
# Ensure the output directory exists
os.makedirs(csv_output_directory, exist_ok=True)

# Collect all CSV files with their metadata
files_metadata = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".csv"):  # Ensure the file is a CSV
            file_path = os.path.join(root, filename)
            parts = root.split(os.sep)  # Split the directory path
            scene = parts[-3]  # Scene type (e.g., NBA, Soccer)
            input_length = parts[-2]  # Input or history length
            model_used = parts[-1]  # Model used (from filename)

            files_metadata.append(
                {
                    "file_path": file_path,
                    "scene": scene,
                    "input_length": input_length,
                    "model_used": model_used,
                }
            )

# Process each file with tqdm
for metadata in tqdm(files_metadata, desc="Processing files"):
    file_path = metadata["file_path"]
    scene = metadata["scene"]
    input_length = metadata["input_length"]
    model_used = metadata["model_used"]

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Process the DataFrame as needed
    times = [25, 50, 75, 100]
    new_rows = []  # List to store new rows
    columns_to_keep = ["Metric", "Mean", "Std"]

    # Create new rows for step-based metrics
    for i in times:
        if f"Error at step {i-1} (mean)" in df.columns:
            error_mean = df[f"Error at step {i-1} (mean)"][0]
            angular_error_mean = df[f"Angular Error at step {i-1} (mean)"][0]
            error_std = df[f"Error at step {i-1} (var)"][0]
            angular_error_std = df[f"Angular Error at step {i-1} (var)"][0]

            # Append new rows to the list with combined Mean and Std
            new_rows.append(
                {
                    "Metric": f"Err {int(i*0.04)}s",
                    "Mean": error_mean,
                    "Std": error_std,
                }
            )
            new_rows.append(
                {
                    "Metric": f"Rad Err {int(i*0.04)}s",
                    "Mean": angular_error_mean,
                    "Std": angular_error_std,
                }
            )

    # Create a DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)

    # Combine original DataFrame with new metrics
    df_combined = pd.concat([df[columns_to_keep], new_df], ignore_index=True)

    # Replace underscores in the DataFrame
    df_combined.columns = [col.replace("_", "\\_") for col in df_combined.columns]
    df_combined["Metric"] = df_combined["Metric"].str.replace("_", "\\_")

    # Replace specific terms
    df_combined["Metric"] = df_combined["Metric"].replace(
        {"FRE (Final Radian Error)": "FRE", "ARE (Average Radian Error)": "ARE"}
    )

    # Combine Mean and Std into Mean_Std
    df_combined["Mean\_Std"] = df_combined.apply(
        lambda row: (
            f"{row['Mean']:.2f} \pm {row['Std']:.2f}"
            if pd.notna(row["Mean"]) and pd.notna(row["Std"])
            else None
        ),
        axis=1,
    )
    df_combined = df_combined.drop(columns=["Mean", "Std"])

    # Generate the filename for the LaTeX file
    latex_filename = f"tables/{scene}_{input_length}_{model_used}.tex"

    # Convert the DataFrame to LaTeX format
    latex_code = df_combined.to_latex(
        index=False, header=True, column_format="lc", float_format="%.2f", escape=False
    )

    # Save the LaTeX output to a file
    with open(latex_filename, "w") as f:
        f.write(latex_code)

    # Generate the filename for the CSV file
    csv_filename = f"{csv_output_directory}/{scene}_{input_length}_{model_used}.csv"

    # Save the DataFrame as a CSV file
    df_combined.to_csv(csv_filename, index=False)

    # Update the progress bar description
    tqdm.write(f"Saved LaTeX table to {latex_filename}")
    tqdm.write(f"Saved processed CSV to {csv_filename}")
