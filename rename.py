import os

def rename_subfolders_recursive(parent_folder):
    # Traverse through the parent folder
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)

        # Check if it's a directory and if its name is '50'
        if os.path.isdir(subfolder_path):
            # Traverse further into subfolder 50
            for nested_subfolder in os.listdir(subfolder_path):
                nested_subfolder_path = os.path.join(subfolder_path, nested_subfolder)

                # If nested subfolder is a directory
                if os.path.isdir(nested_subfolder_path):
                    # Check if 'ostf' is part of the subfolder name
                    if "pbitnet" in nested_subfolder:
                        # Create the new subfolder name by replacing 'ostf' with 'trafo'
                        new_subfolder_name = nested_subfolder.replace("pbitnet", "pos_bitnet")
                        new_subfolder_path = os.path.join(subfolder_path, new_subfolder_name)

                        # Rename the nested subfolder
                        os.rename(nested_subfolder_path, new_subfolder_path)
                        print(f"Renamed {nested_subfolder_path} to {new_subfolder_path}")
        
        # Recurse into other directories to check for the '50' folder
        if os.path.isdir(subfolder_path):
            rename_subfolders_recursive(subfolder_path)

# Example usage:
parent_folder = 'benchmark'  # Adjust to the path where your subfolders are located
rename_subfolders_recursive(parent_folder)
