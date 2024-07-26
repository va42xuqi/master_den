import numpy as np
import matplotlib.pyplot as plt


# Define the functions
def f(x, s):
    return np.tanh(np.abs(x) / s) * np.sign(x)


def g(x, s):
    return np.sign(x) - f(x, s)


# Set the parameter and the range
s = 10
x = np.linspace(-30, 30, 1000)

# Calculate the function values
y_f = f(x, s)
y_g = g(x, s)

# Split the data to avoid the discontinuity at x = 0 for g(x)
x_positive = x[x > 0]
x_negative = x[x < 0]

y_g_positive = g(x_positive, s)
y_g_negative = g(x_negative, s)

# Plot the curves
plt.figure(figsize=(10, 6))
plt.plot(x, y_f)
plt.plot(x_positive, y_g_positive, color="orange")
plt.plot(x_negative, y_g_negative, color="orange")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plots of the functions $f(x)$ and $g(x)$")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.show()
