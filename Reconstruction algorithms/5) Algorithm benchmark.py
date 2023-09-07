import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, dual_annealing
import pickle

# Function definitions from prior work
def even_symmetric_fourier_series(x, *coeffs):
    a0 = coeffs[0]
    result = a0
    for n in range(1, len(coeffs)):
        an = coeffs[n]
        result += an * np.cos(2 * n * np.pi * x / 180)
    return result

def simulate_signal_with_polarization_shift(x_values, fourier_coefficients, weights, polarization_shifts):
    simulated_signal = np.zeros_like(x_values)
    for (m, n), weight, shift in zip(fourier_coefficients.keys(), weights, polarization_shifts):
        coeffs = fourier_coefficients[(m, n)]
        shifted_x = (x_values + shift) % 180
        individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
        simulated_signal += individual_mode_signal
    return simulated_signal

def objective_scalar(params, x, y, fourier_coefficients):
    n_modes = len(fourier_coefficients)
    weights = params[:n_modes]
    shifts = params[n_modes:]
    simulated_y = simulate_signal_with_polarization_shift(x, fourier_coefficients, weights, shifts)
    residual = y - simulated_y
    return np.sum(residual ** 2)

# Define function to plot the original and recovered signals along with their components
def plot_signals(x_values, original_signal, original_components, recovered_signal, recovered_components, sampled_x=None, sampled_y=None):
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
    axes[1].plot(x_values, original_signal, label='Original Signal', linewidth=2)
    if sampled_x is not None and sampled_y is not None:
        axes[1].scatter(sampled_x, sampled_y, label='Sampled Points', color='red', s=50, zorder=5)
    for component, label in recovered_components:
        axes[1].plot(x_values, component, linestyle='--', label=label)
    axes[1].plot(x_values, recovered_signal, label='Recovered Signal', linewidth=2)
    axes[1].set_xlabel('Polarization (degrees)')
    axes[1].set_ylabel('Power (W)')
    axes[1].set_title('Recovered Signal and Components')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Event handler for pick events
def on_pick(event):
    ind = event.ind[0]  # Get the index of the clicked point
    trial_info = trial_data[ind]  # Retrieve the corresponding trial data
    plot_signals(trial_info['x_values'], 
                 trial_info['original_signal'], 
                 trial_info['original_components'], 
                 trial_info['recovered_signal'], 
                 trial_info['recovered_components'],
                 trial_info['sampled_x'],
                 trial_info['sampled_y'])

# Load the data
data = pd.read_csv('Reconstruction algorithms\mode_data_test.csv')
unique_modes = data[['Mode m', 'mode n']].drop_duplicates().values

# Fourier series fit
even_symmetric_fourier_coefficients_corrected = {}
for mode in unique_modes:
    m, n = mode
    mode_data = data[(data['Mode m'] == m) & (data['mode n'] == n)]
    n_terms_adjusted = m + 1
    mode_data = mode_data[mode_data['Polarization'] <= 90]
    x_data = mode_data['Polarization'].values
    y_data = np.nanmean([mode_data['Power1'].values, mode_data['Power2'].values], axis=0)
    popt, _ = curve_fit(even_symmetric_fourier_series, x_data, y_data, p0=[1.0] + [0.0] * n_terms_adjusted)
    even_symmetric_fourier_coefficients_corrected[(m, n)] = popt

# Initialize result storage
avg_percent_diffs = []
all_percent_diffs = []

# Storage for trial data and plots
trial_data = {}

# Initialize result storage
avg_percent_diffs = []
all_percent_diffs = []
trial_count = 0 # Counter variable
trials = 10 # Number of trials per sample

# Loop over number of sample points
for n_points in range(3, 13):
    percent_diffs_for_this_n = []
    for trial in range(trials):
        # Generate random weights and shifts
        original_weights = np.round(np.random.uniform(0, 100, size=len(unique_modes)), 2)
        original_shifts = np.round(np.random.uniform(0, 180, size=len(unique_modes)), 2)
        
        # Generate the original synthetic signal
        x_values = np.linspace(0, 180, 1000)
        original_signal_values = simulate_signal_with_polarization_shift(x_values, even_symmetric_fourier_coefficients_corrected, original_weights, original_shifts)
        
        # Generate the original components for plotting
        original_components = []
        for (m, n), weight, shift in zip(even_symmetric_fourier_coefficients_corrected.keys(), original_weights, original_shifts):
            coeffs = even_symmetric_fourier_coefficients_corrected[(m, n)]
            shifted_x = (x_values + shift) % 180
            individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
            original_components.append((individual_mode_signal, f"{weight}x LP{m}{n} at {shift} deg shift"))

        # Sample the original signal
        sample_indices = np.linspace(0, len(x_values) - 1, n_points, dtype=int)
        sampled_x = x_values[sample_indices]
        sampled_y = original_signal_values[sample_indices]
        
        # Optimization
        bounds = [(0, max(original_weights))] * len(unique_modes) + [(0, 180)] * len(unique_modes)
        result = dual_annealing(objective_scalar, bounds, args=(sampled_x, sampled_y, even_symmetric_fourier_coefficients_corrected), maxiter=100, initial_temp=5230)
        recovered_params = result.x
        recovered_weights = recovered_params[:len(unique_modes)]
        recovered_shifts = recovered_params[len(unique_modes):]
        
        # Generate the recovered signal and its components
        recovered_signal_values = simulate_signal_with_polarization_shift(x_values, even_symmetric_fourier_coefficients_corrected, recovered_weights, recovered_shifts)
        recovered_components = []
        for (m, n), weight, shift in zip(even_symmetric_fourier_coefficients_corrected.keys(), recovered_weights, recovered_shifts):
            coeffs = even_symmetric_fourier_coefficients_corrected[(m, n)]
            shifted_x = (x_values + shift) % 180
            individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
            recovered_components.append((individual_mode_signal, f"{weight:.2f}x LP{m}{n} at {shift:.2f} deg shift"))

        # Store trial data for later plotting
        trial_data[trial_count] = {
            'x_values': x_values,
            'original_signal': original_signal_values,
            'original_components': original_components,
            'recovered_signal': recovered_signal_values,
            'recovered_components': recovered_components,
            'sampled_x': sampled_x,
            'sampled_y': sampled_y
        }
        trial_count += 1

        # Compute percentage differences
        weight_percent_diffs = np.abs((recovered_weights - original_weights) / original_weights) * 100
        shift_percent_diffs = np.abs((recovered_shifts - original_shifts))  # Difference in degrees
        shift_percent_diffs = np.where(shift_percent_diffs > 90, 180 - shift_percent_diffs, shift_percent_diffs)  # Map to [0, 90]
        shift_percent_diffs = (shift_percent_diffs / 90) * 100  # Convert to percentage
        
        # Calculate total average percentage difference for this trial
        total_percent_diff = np.mean(np.concatenate([weight_percent_diffs, shift_percent_diffs]))
        percent_diffs_for_this_n.append(total_percent_diff)
    
    # Calculate and store the average percentage difference for this n_points
    avg_percent_diff = np.mean(percent_diffs_for_this_n)
    avg_percent_diffs.append(avg_percent_diff)
    all_percent_diffs.extend(percent_diffs_for_this_n)

# Re-run the plotting with pick event
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(np.repeat(np.arange(3, 13), trials), all_percent_diffs, c='blue', alpha=0.5, picker=True)
ax.plot(np.arange(3, 13), avg_percent_diffs, c='red', marker='o')
ax.set_xlabel('Number of Sample Points')
ax.set_ylabel('Total Average % Difference')
ax.set_title('Total Average % Difference vs Number of Sample Points')
ax.grid(True)

fig.canvas.callbacks.connect('pick_event', on_pick)
plt.show()
with open("Reconstruction algorithms\Algorithm_Benchmark_Figure.pkl", "wb") as f:
    pickle.dump(fig, f)