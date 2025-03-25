import matplotlib.pyplot as plt
import numpy as np

models = ['LSTM', 'LMU', 'BitNet', 'Transformer']
FDE = [16.91, 17.47, 18.60, 16.66]
params = [3.581, 1.213, 19.541, 19.430]  # Million parameters
energy = [33.53, 141.85, 324.54, 206.02]  # Energy in milliWatt-hours

# Plot 1: FDE vs Params (Model Complexity in Million Parameters)
plt.figure(figsize=(8,6))
plt.scatter(params, FDE, color='blue', s=100, marker='x', linewidths=2)
plt.xlabel('Model Complexity (Model Size in MB)')
plt.ylabel('FDE (cm)')
plt.title('FDE Performance vs Model Complexity (Params)')

# Ensure x-axis is linear (default), and set ticks properly
plt.xscale('linear')
plt.xticks(np.arange(0, max(params) + 1, 5))  # Customize the ticks with a range and step size

# Adding labels further away from the points
for i, model in enumerate(models):
    plt.text(params[i] + 0.1, FDE[i], f'{model}', fontsize=12, ha='left')

plt.grid(True)
plt.savefig('FDE_vs_Params.png')

# Plot 2: FDE vs Energy (Model Complexity in Energy in milliWatt-hours)
plt.figure(figsize=(8,6))
plt.scatter(energy, FDE, color='green', s=100, marker='x', linewidths=2)
plt.xlabel('Model Complexity (Energy in milliWatt-hours)')
plt.ylabel('FDE (cm)')
plt.title('FDE Performance vs Model Complexity (Energy)')

# Set x-axis to linear scale and custom ticks for better readability
plt.xscale('linear')
plt.xticks(np.arange(0, max(energy) + 50, 50))  # Customize the ticks with a range and step size

# Adding labels further away from the points
for i, model in enumerate(models):
    plt.text(energy[i] + 10, FDE[i], f'{model}', fontsize=12, ha='left')

plt.grid(True)
plt.savefig('FDE_vs_Energy.png')
