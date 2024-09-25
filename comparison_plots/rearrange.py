import os
import numpy as np
import pandas as pd

def collect_and_save_data(root_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Traverse through each subfolder (structured as /scene/input_length/model/)
    for scene in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        
        for input_length in os.listdir(scene_path):
            input_length_path = os.path.join(scene_path, input_length)
            if not os.path.isdir(input_length_path):
                continue
            
            for model in os.listdir(input_length_path):
                model_path = os.path.join(input_length_path, model)
                if not os.path.isdir(model_path):
                    continue

                # Construct the full path to each file
                angular_mean_path = os.path.join(model_path, "angular_mean.npy")
                angular_std_path = os.path.join(model_path, "angular_var.npy")
                error_mean_path = os.path.join(model_path, "error_mean.npy")
                error_std_path = os.path.join(model_path, "error_var.npy")

                if os.path.exists(angular_mean_path) and os.path.exists(angular_std_path) \
                        and os.path.exists(error_mean_path) and os.path.exists(error_std_path):
                    
                    # Load the data from .npy files
                    angular_mean = np.load(angular_mean_path)
                    angular_std = np.load(angular_std_path)
                    error_mean = np.load(error_mean_path)
                    error_std = np.load(error_std_path)

                    # Convert values greater than 20 to radians
                    angular_mean = np.where(angular_mean > 20, np.radians(angular_mean), angular_mean)
                    angular_std = np.where(angular_std > 20, np.radians(angular_std), angular_std)
                    error_mean = np.where(error_mean > 20, np.radians(error_mean), error_mean)
                    error_std = np.where(error_std > 20, np.radians(error_std), error_std)

                    # Combine into a DataFrame
                    combined_data = {
                        'angular_mean': angular_mean.flatten(),
                        'angular_std': angular_std.flatten(),  # Treat var as std as mentioned
                        'error_mean': error_mean.flatten(),
                        'error_std': error_std.flatten(),  # Treat var as std as mentioned
                    }
                    
                    df = pd.DataFrame(combined_data)

                    # Save the combined data into the flattened directory structure
                    save_name = f"{scene}_{input_length}_{model}_combined.csv"
                    save_path = os.path.join(save_dir, save_name)
                    df.to_csv(save_path, index=False)

# Example usage
root_directory = "../benchmark"  # Replace with your root directory path
save_directory = "./benchmark"  # Replace with your desired save directory path
collect_and_save_data(root_directory, save_directory)
