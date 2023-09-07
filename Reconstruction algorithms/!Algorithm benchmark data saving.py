import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, dual_annealing
import os
import time

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
    if original_components is not None:
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
    if recovered_components is not None:
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

    # Load the trial data from the corresponding CSV file
    df_loaded = pd.read_csv(f"{output_folder}trial_{ind}.csv")
    x_values = df_loaded['x_values'].values
    original_signal = df_loaded['original_signal'].values
    recovered_signal = df_loaded['recovered_signal'].values
    sampled_x = df_loaded['sampled_x'].dropna().values
    sampled_y = df_loaded['sampled_y'].dropna().values

    # Determine the number of original and recovered components based on column names
    original_component_cols = [col for col in df_loaded.columns if 'original_component_' in col]
    recovered_component_cols = [col for col in df_loaded.columns if 'recovered_component_' in col]
    number_of_original_components = len(original_component_cols)
    number_of_recovered_components = len(recovered_component_cols)

    # Load original and recovered components
    original_components = []
    recovered_components = []
    
    for i in range(number_of_original_components):
        component = df_loaded[f'original_component_{i}'].values
        original_components.append(component)
        
    for i in range(number_of_recovered_components):
        component = df_loaded[f'recovered_component_{i}'].values
        recovered_components.append(component)
    
    # Load the labels from the corresponding text file
    with open(f"{output_folder}trial_{ind}_labels.txt", "r") as label_file:
        content = label_file.read().split("\n")
        original_start = content.index("Original Labels:") + 1
        recovered_start = content.index("Recovered Labels:") + 1

        original_labels = content[original_start:recovered_start-1]
        recovered_labels = content[recovered_start:]

    # Pair the loaded components with their corresponding labels
    original_components = [(component, label) for component, label in zip(original_components, original_labels)]
    recovered_components = [(component, label) for component, label in zip(recovered_components, recovered_labels)]

    plot_signals(x_values, 
                 original_signal, 
                 original_components, 
                 recovered_signal, 
                 recovered_components,
                 sampled_x,
                 sampled_y)

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
output_folder = "Reconstruction algorithms\Benchmark_data/"

# Initialize result storage
avg_percent_diffs = []
all_percent_diffs = []
trial_count = 0 # Counter variable
trials = 50 # Number of trials per sample
total_trials = (13 - 3) * trials #TO BE REPLACED WITH RANGE VARIABLES
start_time = time.time()
bar_length = 100

# Delete all files in the output folder
for filename in os.listdir(output_folder):
    file_path = os.path.join(output_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Failed to delete {file_path}. Reason: {e}")

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

        # Save the trial data as a CSV file
        df_to_save = pd.DataFrame({
            'x_values': x_values,
            'original_signal': original_signal_values,
            'recovered_signal': recovered_signal_values,
            'sampled_x': np.concatenate([sampled_x, [np.nan] * (len(x_values) - len(sampled_x))]),
            'sampled_y': np.concatenate([sampled_y, [np.nan] * (len(x_values) - len(sampled_y))])
        })
        # Add original and recovered components
        for i, (component, label) in enumerate(original_components):
            df_to_save[f'original_component_{i}'] = component
        
        for i, (component, label) in enumerate(recovered_components):
            df_to_save[f'recovered_component_{i}'] = component

        # Save the DataFrame as a CSV
        df_to_save.to_csv(f"{output_folder}trial_{trial_count}.csv", index=False)

        original_labels = [label for _, label in original_components]
        recovered_labels = [label for _, label in recovered_components]

        with open(f"{output_folder}trial_{trial_count}_labels.txt", "w") as label_file:
            label_file.write("Original Labels:\n")
            label_file.write("\n".join(original_labels))
            label_file.write("\nRecovered Labels:\n")
            label_file.write("\n".join(recovered_labels))

        # Compute percentage differences
        weight_percent_diffs = np.abs((recovered_weights - original_weights) / original_weights) * 100
        shift_percent_diffs = np.abs((recovered_shifts - original_shifts))  # Difference in degrees
        shift_percent_diffs = np.where(shift_percent_diffs > 90, 180 - shift_percent_diffs, shift_percent_diffs)  # Map to [0, 90]
        shift_percent_diffs = (shift_percent_diffs / 90) * 100  # Convert to percentage
        
        # Calculate total average percentage difference for this trial
        total_percent_diff = np.mean(np.concatenate([weight_percent_diffs, shift_percent_diffs]))
        percent_diffs_for_this_n.append(total_percent_diff)

        # Calculate elapsed and estimated remaining time
        elapsed_time = time.time() - start_time
        estimated_remaining_time = (elapsed_time / trial_count) * (total_trials - trial_count)

        # Update the progress bar
        progress = (trial_count / total_trials)
        arrow = '=' * int(round(progress * bar_length) - 1)
        spaces = ' ' * (bar_length - len(arrow))

        # Print the progress bar
        print(f"[{arrow + spaces}] {progress * 100:.2f}% - Elapsed: {elapsed_time:.2f}s - Remaining: {estimated_remaining_time:.2f}s", end='\r')

    
    # Calculate and store the average percentage difference for this n_points
    avg_percent_diff = np.mean(percent_diffs_for_this_n)
    avg_percent_diffs.append(avg_percent_diff)
    all_percent_diffs.extend(percent_diffs_for_this_n)

print("\nCompleted.")

# Save the summary data to a CSV file in the output folder
pd.DataFrame({'avg_percent_diffs': avg_percent_diffs}).to_csv(f"{output_folder}avg_percent_diffs.csv", index=False)
pd.DataFrame({'all_percent_diffs': all_percent_diffs}).to_csv(f"{output_folder}all_percent_diffs.csv", index=False)


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