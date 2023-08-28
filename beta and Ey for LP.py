from scipy.special import jn_zeros
import numpy as np

def propagation_constant(frequency, a, m, n):
    """
    Calculate the propagation constant for a given LP_mn mode.
    
    Parameters:
    - frequency: Frequency in Hz
    - a: Waveguide radius in meters
    - m: Mode index m
    - n: Mode index n
    
    Returns:
    - beta_mn: Propagation constant in rad/m for the given LP_mn mode
    """
    # Speed of light in vacuum (m/s)
    c = 2.99792458e8  
    
    # Wave number
    k = 2 * np.pi * frequency / c
    
    # nth zero of the Bessel function of order m
    X_mn = jn_zeros(m, n)[-1]
    
    # Propagation constant for LP_mn mode
    beta_mn = np.sqrt(k**2 - ((X_mn/a)**2))
    
    return beta_mn, X_mn

def Ey_expression(a, X_mn, m, polarization='odd'):
    """
    Generate the E_y expression for COMSOL for a given LP_mn mode.
    
    Parameters:
    - a: Waveguide radius in meters
    - X_mn: nth zero of the Bessel function of order m
    - m: Mode index m
    - polarization: 'odd' for cosine and 'even' for sine polarization
    
    Returns:
    - Ey_mn_expression: Expression for E_y in COMSOL syntax
    """
    r_expression = "sqrt(x*x + z*z)"
    phi_expression = "atan2(z, x)"
    
    # Mapping the input to the respective trigonometric function
    trig_function = 'cos' if polarization == 'odd' else 'sin'
    
    # Radial dependency
    radial_dependency = f"besselj({m}, {X_mn/a} [m^-1] * {r_expression})"
    
    # Azimuthal dependency
    azimuthal_dependency = f"{trig_function}({m} * {phi_expression})"
    
    # Combine both dependencies
    Ey_mn_expression = f"{radial_dependency} * {azimuthal_dependency}"
    
    return Ey_mn_expression

# Calculate for LP_01 mode at 170 GHz with a radius of 8.82 mm
frequency = 170e9
a = 7.01668E-3
m, n = 4, 4

beta_01, X_01 = propagation_constant(frequency, a, m, n)
Ey_01_expression = Ey_expression(a, X_01, m, 'odd')

print(f"{beta_01} rad/m")
print(Ey_01_expression)