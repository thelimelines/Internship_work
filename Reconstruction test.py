# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

# Load the data
data = pd.read_csv('mode_data_test.csv')

# Define the even symmetric Fourier series function
def even_symmetric_fourier_series(x, *coeffs):
    a0 = coeffs[0]
    result = a0
    for n in range(1, len(coeffs)):
        an = coeffs[n]
        result += an * np.cos(2 * n * np.pi * x / 180)
    return result

# Identify unique modes in the data
unique_modes = data[['Mode m', 'mode n']].drop_duplicates().values

# Initialize a dictionary to store Fourier coefficients for each mode
even_symmetric_fourier_coefficients = {}

# Fit the even symmetric Fourier series for each unique mode using data in the range [0, 90]
for mode in unique_modes:
    m, n = mode
    mode_data = data[(data['Mode m'] == m) & (data['mode n'] == n)]
    
    # Adjust the number of terms based on m
    n_terms_adjusted = m + 1
    
    # Filter data for x in [0, 90]
    mode_data = mode_data[mode_data['Polarization'] <= 90]
    
    x_data = mode_data['Polarization'].values
    y_data = np.nanmean([mode_data['Power1'].values, mode_data['Power2'].values], axis=0)
    
    # Fit the Fourier series to the data with adjusted number of terms
    popt, _ = curve_fit(even_symmetric_fourier_series, x_data, y_data, p0=[1.0] + [0.0] * n_terms_adjusted)
    even_symmetric_fourier_coefficients[(m, n)] = popt

# Function to simulate signal with polarization shift
def simulate_signal_with_polarization_shift(x_values, fourier_coefficients, weights, polarization_shifts):
    simulated_signal = np.zeros_like(x_values)
    for (m, n), weight, shift in zip(fourier_coefficients.keys(), weights, polarization_shifts):
        coeffs = fourier_coefficients[(m, n)]
        shifted_x = (x_values + shift) % 180
        individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
        simulated_signal += individual_mode_signal
    return simulated_signal

# Generate the simulated signal
x_values = np.linspace(0, 180, 1000)
weights = [1, 2, 3]
polarization_shifts = [0, 30, 15]
simulated_signal_values = simulate_signal_with_polarization_shift(x_values, even_symmetric_fourier_coefficients, weights, polarization_shifts)

# Perform sampling at n_points equally spaced points
n_points = 6
sample_indices = np.linspace(0, len(x_values) - 1, n_points, dtype=int)
sampled_x = x_values[sample_indices]
sampled_y = simulated_signal_values[sample_indices]

# Function for optimization to recover modes
def objective(params, x, y, fourier_coefficients):
    n_modes = len(fourier_coefficients)
    weights = params[:n_modes]
    shifts = params[n_modes:]
    simulated_y = simulate_signal_with_polarization_shift(x, fourier_coefficients, weights, shifts)
    return y - simulated_y

# Initial guess for optimization (weights and shifts)
initial_guess = np.concatenate([np.ones(len(unique_modes)), np.zeros(len(unique_modes))])

# Perform optimization to recover modes
result = least_squares(objective, initial_guess, args=(sampled_x, sampled_y, even_symmetric_fourier_coefficients))
recovered_params = result.x
recovered_weights = recovered_params[:len(unique_modes)]
recovered_shifts = recovered_params[len(unique_modes):]

# Generate the recovered signal using the optimized parameters
recovered_signal_values = simulate_signal_with_polarization_shift(x_values, even_symmetric_fourier_coefficients, recovered_weights, recovered_shifts)

# Function for plotting the original and recovered signals along with their components
def plot_signals(x_values, original_signal, original_components, recovered_signal, recovered_components, sampled_points=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot the original simulated signal and its components
    axes[0].plot(x_values, original_signal, label='Original Signal', linewidth=2)
    for component, label in original_components:
        axes[0].plot(x_values, component, linestyle='--', label=label)
    axes[0].set_xlabel('Polarization (degrees)')
    axes[0].set_ylabel('Power (W)')
    axes[0].set_title('Original Signal and Components')
    axes[0].legend()
    
    # Plot the recovered signal and its components
    axes[1].plot(x_values, recovered_signal, label='Recovered Signal', linewidth=2)
    if sampled_points is not None:
        axes[1].scatter(sampled_points[0], sampled_points[1], label='Sampled Points', color='red', s=50, zorder=5)
    for component, label in recovered_components:
        axes[1].plot(x_values, component, linestyle='--', label=label)
    axes[1].set_xlabel('Polarization (degrees)')
    axes[1].set_ylabel('Power (W)')
    axes[1].set_title('Recovered Signal and Components')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Generate the original components for plotting
original_components = []
for (m, n), weight, shift in zip(even_symmetric_fourier_coefficients.keys(), weights, polarization_shifts):
    coeffs = even_symmetric_fourier_coefficients[(m, n)]
    shifted_x = (x_values + shift) % 180
    individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
    original_components.append((individual_mode_signal, f"{weight}x LP{m}{n} at {shift} deg shift"))

# Generate the recovered components for plotting
recovered_components = []
for (m, n), weight, shift in zip(even_symmetric_fourier_coefficients.keys(), recovered_weights, recovered_shifts):
    coeffs = even_symmetric_fourier_coefficients[(m, n)]
    shifted_x = (x_values + shift) % 180
    individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
    recovered_components.append((individual_mode_signal, f"{weight:.2f}x LP{m}{n} at {shift:.2f} deg shift"))

# Sampled points for plotting
sampled_points = (sampled_x, sampled_y)

# Plot the signals and their components along with the sampled points
plot_signals(x_values, simulated_signal_values, original_components, recovered_signal_values, recovered_components, sampled_points=sampled_points)


# Output the exact recovered weights and polarization shifts for each mode
recovered_mode_info = {tuple(mode): {'Weight': weight, 'Polarization Shift': shift} for mode, weight, shift in zip(unique_modes, recovered_weights, recovered_shifts)}