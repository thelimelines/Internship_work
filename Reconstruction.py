import numpy as np
from scipy.optimize import curve_fit

# Define the even symmetric Fourier series function
def even_symmetric_fourier_series(x, *coeffs):
    a0 = coeffs[0]
    result = a0
    for n in range(1, len(coeffs)):
        an = coeffs[n]
        result += an * np.cos(2 * n * np.pi * x / 180)
    return result

# Fourier coefficients obtained from previous curve fitting (for LP01 and LP11 modes)
fourier_coefficients = {
    (0, 1): [0.4, 0.3],  # Coefficients for LP01
    (1, 1): [0.2, 0.1]  # Coefficients for LP11
}

# Define the new setup: 4 holes and 2 modes
new_holes = np.array([0, 20, 30, 70])  # Angles in degrees
new_modes = [(0, 1), (1, 1)]  # LP01 and LP11
new_modes_powers = np.array([0.9, 0.1])  # Powers in fractions
new_modes_orientations = np.array([0, 45])  # Orientations in degrees

# Initialize the new system matrix A (4 holes x 2 modes)
new_A_matrix = np.zeros((len(new_holes), len(new_modes)))

# Fill the new system matrix A
for i, angle in enumerate(new_holes):
    for j, mode in enumerate(new_modes):
        basis_fn_value = even_symmetric_fourier_series(angle, *fourier_coefficients[mode])
        new_A_matrix[i, j] = basis_fn_value

# Normalize the new system matrix rows
row_sums_new = new_A_matrix.sum(axis=1)
new_A_matrix = new_A_matrix / row_sums_new[:, np.newaxis]

# Generate synthetic observations (new b vector)
new_b_vector = np.zeros(len(new_holes))
for i, angle in enumerate(new_holes):
    for j, (mode, power, orientation) in enumerate(zip(new_modes, new_modes_powers, new_modes_orientations)):
        contribution = power * even_symmetric_fourier_series(angle - orientation, *fourier_coefficients[mode])
        new_b_vector[i] += contribution

# Normalize the new b vector
new_b_vector = new_b_vector / np.sum(new_b_vector)

# Solve the inversion problem (Ax = b) to find the new x vector
new_A_inv = np.linalg.pinv(new_A_matrix)
new_x_vector = np.dot(new_A_inv, new_b_vector)

# Calculate the power fractions based on the new x vector
total_power_new = np.sum(new_x_vector)
power_fractions_new = new_x_vector / total_power_new

new_A_matrix, new_b_vector, new_x_vector, power_fractions_new
