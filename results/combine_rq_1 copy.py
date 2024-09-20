import os
import pandas as pd

# Directory containing the CSV files
directory = "results/tables"
# Directory to save the CSV files
csv_output_directory = "results/combined"
# Directory to save the LaTeX files
latex_output_directory = "tables/combined"
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
                if "50" in filename and scene_type in filename:
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
        model_name = model_name.split("50_")[1]
        model_name = model_name.replace("os", "")
        model_name = model_name.replace("_", " ")
        if model_name == "tf":
            model_name = "Transformer"
        if model_name == "bitnet":
            model_name = "BitNet"
        # Capitalize each word in the model name
        if model_name in ["lstm","lmu"]:
            model_name = model_name.upper()
        df["Model"] = model_name
        # Append DataFrame to the list
        data_frames.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Pivot the DataFrame to have models as columns
    pivot_df = combined_df.pivot(
        index="Metric", columns="Model", values="Mean\_Std"
    ).reset_index()

    # Split the pivoted DataFrame into two parts
    mid_point = len(pivot_df.columns) // 2 + 1
    first_part = pivot_df.iloc[:, :mid_point]
    second_part = pivot_df.iloc[:, [0] + list(range(mid_point, len(pivot_df.columns)))]

    # Save each part to a separate CSV file
    csv_file_path_1 = os.path.join(
        csv_output_directory, f"pivoted_data_50_{scene_type}_part1.csv"
    )
    first_part.to_csv(csv_file_path_1, index=False)

    csv_file_path_2 = os.path.join(
        csv_output_directory, f"pivoted_data_50_{scene_type}_part2.csv"
    )
    second_part.to_csv(csv_file_path_2, index=False)

    print(f"CSV files for {scene_type} saved to {csv_file_path_1} and {csv_file_path_2}")

    # Convert each part to LaTeX format with longtable
    latex_table_1 = first_part.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l" + "c" * (first_part.shape[1] - 1),
        caption=f"Results for {scene_type} scene type (Part 1).",
        label=f"{scene_type} results part1",
        escape=False  # To handle special characters if needed
    )
    latex_table_2 = second_part.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l" + "c" * (second_part.shape[1] - 1),
        caption=f"Results for {scene_type} scene type (Part 2).",
        label=f"{scene_type} results part2",
        escape=False  # To handle special characters if needed
    )

    latex_table_1 = latex_table_1.replace("\\begin{table}", "\\begin{table}[H]\n\\centering")
    latex_table_2 = latex_table_2.replace("\\begin{table}", "\\begin{table}[H]\n\\centering")

    # Save each LaTeX table to a separate .tex file
    latex_file_path_1 = os.path.join(
        latex_output_directory, f"pivoted_data_50_{scene_type}_part1.tex"
    )
    with open(latex_file_path_1, "w") as f:
        f.write(latex_table_1)

    latex_file_path_2 = os.path.join(
        latex_output_directory, f"pivoted_data_50_{scene_type}_part2.tex"
    )
    with open(latex_file_path_2, "w") as f:
        f.write(latex_table_2)

    print(f"LaTeX tables for {scene_type} saved to {latex_file_path_1} and {latex_file_path_2}")


# Process both SOC and NBA scene types
process_files("SOC")
process_files("NBA")
