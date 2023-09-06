import numpy as np
from scipy.special import jn_zeros

def mode_wavelength(f, a, m, n):
    # Constants
    c = 2.99792458e8  # speed of light in m/s
    
    # Wave number in free space based on frequency
    k = 2 * np.pi * f / c
    
    # Cut-off wave numbers for the modes using Bessel function zeros
    kc_mn = jn_zeros(m, n)[-1] / a
    lambdac_mn = jn_zeros(m, n)[-1] / k

    # Propagation constants for each mode
    beta_mn = np.sqrt(k**2 - kc_mn**2)
    
    # Calculate the wavelength
    mode_lambda = 2 * np.pi / np.abs(beta_mn)
    
    result_string = f"The wavelength for LP{m}{n} is {mode_lambda:.6f} m with a hole cutoff radius of {lambdac_mn*1000} mm"
    return result_string

# Provided values
f = 170e9  # 170 GHz
a = 7.01668E-3  # radius in meters
m, n = 0,1

# Calculate and print the beat wavelength for LP01 and LP11
wavelength_result = mode_wavelength(f, a, m, n)
print(wavelength_result)