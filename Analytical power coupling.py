import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jn

# Function to compute the propagation constant for a given mode in a circular corrugated waveguide
def propagation_constant(frequency, a, m, n):
    c = 2.99792458e8
    k = 2 * np.pi * frequency / c  # Wave number in free space
    X_mn = jn_zeros(m, n)[-1]  # Bessel function zero for the given mode
    beta_mn = np.sqrt(k**2 - ((X_mn/a)**2))  # Propagation constant
    return beta_mn, X_mn

# Function to compute the electric field along a chord for a given azimuthal angle
def Ey_along_chord(phi, chord_length, X_mn, m, a):
    r = a
    phi_start, phi_end = get_chord_angles(phi, a, chord_length)  # Start and end angles of the chord
    phi_values = np.linspace(phi_start, phi_end, 1000)  # Generating angles for the chord
    # Computing electric field values along the chord
    Ey_values = [jn(m, X_mn * r) * np.cos(m * phi_val) for phi_val in phi_values]
    return phi_values, Ey_values

# Function to get the start and end angles for a chord given its midpoint azimuthal angle
def get_chord_angles(phi, a, chord_length):
    delta_phi = np.arcsin(chord_length / (2 * a))  # Change in azimuthal angle due to chord
    return phi - delta_phi, phi + delta_phi  # Return start and end angles

# Function to compute the power for a given azimuthal angle after including the evanescent decay
def compute_power_for_phi_final(phi, X_mn, m, a, chord_length, P_total, Ey_max, beta, decay_distance):
    phi_values, Ey_values = Ey_along_chord(phi, chord_length, X_mn, m, a)  # Get electric field values along the chord
    # Normalize the electric field values
    Ey_normal_values = [Ey_val * np.cos(phi_val - phi) * np.cos(phi) for phi_val, Ey_val in zip(phi_values, Ey_values)]
    decay_factor = np.exp(-beta * decay_distance)  # Evanescent decay factor
    # Apply the decay factor to the electric field values
    Ey_normal_values_decayed = [Ey_val * decay_factor for Ey_val in Ey_normal_values]
    # Integrate the electric field to get the total field
    total_field = np.trapz(Ey_normal_values_decayed, x=phi_values)
    # Compute the power for the hole using the total field and given total power
    P_hole = P_total * (total_field / Ey_max)**2
    return P_hole

# Parameters for the calculations
frequency = 170e9  # Frequency of the wave
a = 7.01668E-3  # Radius of the waveguide
m, n = 4, 4  # Mode numbers
chord_length = 1e-3  # Length of the chord in the waveguide (hole size)
P_total = 1e6  # Total power in the waveguide
phi_values_refined_range = np.linspace(0, np.pi, 500)  # Azimuthal angles for which power will be computed
decay_distance = 1.43e-3  # Distance over which evanescent decay occurs

# Calculations
beta_mn, X_mn = propagation_constant(frequency, a, m, n)  # Compute propagation constant and Bessel function zero
# Compute the maximum electric field value at the waveguide edge as an approximation
Ey_max = jn(m, X_mn * a) * np.cos(m * np.pi/4)  
# Compute the power for each azimuthal angle
P_hole_values_final = [compute_power_for_phi_final(phi, X_mn, m, a, chord_length, P_total, Ey_max, beta_mn, decay_distance) 
                       for phi in phi_values_refined_range]

# Plotting the computed power values against the azimuthal angles
plt.figure(figsize=(10,6))
plt.plot(phi_values_refined_range, P_hole_values_final)
plt.xlabel('Phi (radians)')
plt.ylabel('Power coupling (W)')
plt.title(f'Final Power coupling (with evanescent decay) for LP_{m}{n} mode')
plt.grid(True)
plt.show()
