import numpy as np
from scipy.special import jn_zeros

def beat_wavelength_with_frequency(f, a, m1, n1, m2, n2):
    # Constants
    c = 2.99792458e8  # speed of light in m/s
    
    # Wave number in free space based on frequency
    k = 2 * np.pi * f / c
    
    # Cut-off wave numbers for the modes using Bessel function zeros
    kc_m1n1 = jn_zeros(m1, n1)[-1] / a
    kc_m2n2 = jn_zeros(m2, n2)[-1] / a
    
    # Propagation constants for each mode
    beta_m1n1 = np.sqrt(k**2 - kc_m1n1**2)
    beta_m2n2 = np.sqrt(k**2 - kc_m2n2**2)
    
    # Calculate the beat wavelength
    lambda_B = 2 * np.pi / np.abs(beta_m1n1 - beta_m2n2)
    
    result_string = f"The beat wavelength between LP{m1}{n1} mode and LP{m2}{n2} mode is {lambda_B:.6f} m"
    return result_string

# Provided values
f = 170e9  # 170 GHz in Hz
a = 17.64e-3  # 17.64 mm in meters
m1, n1 = 0, 1
m2, n2 = 1, 1

# Calculate and print the beat wavelength for LP01 and LP11
beat_wavelength_result = beat_wavelength_with_frequency(f, a, m1, n1, m2, n2)
print(beat_wavelength_result)