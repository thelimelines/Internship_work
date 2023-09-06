# Import the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data = pd.read_csv('Reconstruction algorithms\mode_data_test.csv')

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
    n_terms_adjusted = m + 1
    mode_data = mode_data[mode_data['Polarization'] <= 90]
    x_data = mode_data['Polarization'].values
    y_data = np.nanmean([mode_data['Power1'].values, mode_data['Power2'].values], axis=0)
    popt, _ = curve_fit(even_symmetric_fourier_series, x_data, y_data, p0=[1.0] + [0.0] * n_terms_adjusted)
    even_symmetric_fourier_coefficients[(m, n)] = popt

# Plot the results
plt.figure(figsize=(15, 12))
for idx, mode in enumerate(unique_modes, 1):
    m, n = mode
    x_vals = np.linspace(0, 180, 1000)
    y_vals_original = even_symmetric_fourier_series(x_vals, *even_symmetric_fourier_coefficients[(m, n)])
    
    # Manually shift the array by 90 degrees
    shifted_indices = int(len(x_vals) * 90 / 180)  # Indices to shift corresponding to 90 degrees
    y_vals_shifted = np.roll(y_vals_original, shifted_indices)
    
    # Calculate the sum of the Fourier series fit values and their shifted counterparts
    sum_vals = y_vals_original + y_vals_shifted
    
    # Plot data
    plt.subplot(len(unique_modes), 1, idx)
    plt.plot(x_vals, y_vals_original, label='Original Fit', color='red')
    plt.plot(x_vals, y_vals_shifted, label='90-degree Shifted Fit', color='blue')
    plt.plot(x_vals, sum_vals, label='Sum of Fits', color='green')
    
    term_string = "term" if m == 0 else "terms"
    plt.title(f"Mode LP{m}{n} with {m + 1} {term_string}")
    plt.xlabel('Polarization (degrees)')
    plt.ylabel('Power (W)')
    plt.legend()

plt.tight_layout()
plt.show()
