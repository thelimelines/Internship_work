# Import the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data = pd.read_csv('Reconstruction algorithms\mode_data_test.csv')

# Define the even symmetric Fourier series function
def even_symmetric_fourier_series(x, *coeffs):
    """Even symmetric Fourier series representation using only cosine terms."""
    a0 = coeffs[0]
    result = a0
    for n in range(1, len(coeffs)):
        an = coeffs[n]
        result += an * np.cos(2 * n * np.pi * x / 180)
    return result

# Generate equations for each mode's fit
def generate_equation(coeffs):
    """Generate the equation string for a given set of Fourier coefficients."""
    equation = f"y(x) = {coeffs[0]:.3f}"
    for n, coeff in enumerate(coeffs[1:], 1):
        equation += f" + {coeff:.3f}cos({2*n}x)"
    return equation

# Identify unique modes in the data
unique_modes = data[['Mode m', 'mode n']].drop_duplicates().values

# Initialize a dictionary to store Fourier coefficients for each mode
even_symmetric_fourier_coefficients = {}
equations = {}

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

    # Generate and store the equation for this mode
    equations[(m, n)] = generate_equation(popt)

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
