import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from Reconstruction_module import even_symmetric_fourier_series, simulate_signal_with_polarization_shift, objective_scalar, even_symmetric_fourier_coefficients_corrected

# Initialize variables to hold the percentage differences
percent_diffs = []

# Number of random sets
n_sets = 1000

# Loop through each random set
for _ in range(n_sets):
    # Generate random amplitudes and polarizations
    random_weights = np.random.uniform(0, 100, 3)
    random_shifts = np.random.uniform(0, 180, 3)

    # Generate synthetic signal
    x_values = np.linspace(0, 180, 1000)
    original_signal = simulate_signal_with_polarization_shift(x_values, even_symmetric_fourier_coefficients_corrected, random_weights, random_shifts)

    # Initialize list to hold percentage differences for each sample size
    sample_size_diffs = []
  
    # Loop through different sample sizes
    for n_points in range(2, 21):  # From 2 to 20 samples
        # Sample the original signal
        sample_indices = np.linspace(0, len(x_values) - 1, n_points, dtype=int)
        sampled_x = x_values[sample_indices]
        sampled_y = original_signal[sample_indices]

        # Set the bounds for the optimization
        bounds = [(0, 10**10)] * 3 + [(0, 180)] * 3

        # Perform optimization
        result = dual_annealing(objective_scalar, bounds, args=(sampled_x, sampled_y, even_symmetric_fourier_coefficients_corrected))
        recovered_params = result.x
        recovered_weights = recovered_params[:3]
        recovered_shifts = recovered_params[3:]

        # Calculate the percentage differences
        weight_diffs = np.abs((recovered_weights - random_weights) / random_weights) * 100
        shift_diffs = np.abs((recovered_shifts - random_shifts) / random_shifts) * 100

        # Calculate the average percentage difference for this sample size
        avg_diff = np.mean(np.concatenate([weight_diffs, shift_diffs]))
        sample_size_diffs.append(avg_diff)

    # Append to the list holding percentage differences for all sets
    percent_diffs.append(sample_size_diffs)

# Calculate the mean and standard deviation of the percentage differences for each sample size
mean_diffs = np.mean(percent_diffs, axis=0)
std_diffs = np.std(percent_diffs, axis=0)

# Plotting
plt.errorbar(range(2, 21), mean_diffs, yerr=std_diffs, fmt='-o')
plt.xlabel('Number of Samples')
plt.ylabel('Average % Difference')
plt.title('Accuracy of Signal Reconstruction vs. Number of Samples')
plt.show()
