import numpy as np

# Replace 'your_file.npy' with the actual file path
file_path = 'benchmark/NBA/50/ostf/angular_mean.npy'

# Load the .npy file
data = np.load(file_path)

print(data)

# Display the data
mean = np.mean(data[:50])

print(mean/np.pi*180)