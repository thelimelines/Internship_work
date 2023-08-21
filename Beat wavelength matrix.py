import pandas as pd
import numpy as np
from scipy.special import jn_zeros

def compute_beat_wavelength(f, a, m1, n1, m2, n2):
    """
    Computes the beat wavelength between two modes given their m and n values, frequency, and guide radius.
    """
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

    
    return lambda_B

def modified_generate_wavelength_dataframe(f, a, m_values, n_values):
    """
    Generates a DataFrame containing beat wavelengths between different modes.
    """
    # Create an empty DataFrame
    df = pd.DataFrame(index=[f"LP{m}{n}" for m in m_values for n in n_values], 
                      columns=[f"LP{m}{n}" for m in m_values for n in n_values])

    # Fill the DataFrame with beat wavelengths
    for m1 in m_values:
        for n1 in n_values:
            for m2 in m_values:
                for n2 in n_values:
                    if (m1, n1) != (m2, n2):  # Skip the same mode combinations
                        lambda_B = compute_beat_wavelength(f, a, m1, n1, m2, n2)
                        df.at[f"LP{m1}{n1}", f"LP{m2}{n2}"] = lambda_B
                    elif (m1, n1) == (m2, n2):  # For diagonal set NaN
                        df.at[f"LP{m1}{n1}", f"LP{m2}{n2}"] = np.nan

    # Mask the lower triangle of the DataFrame to retain only upper triangle values
    df_upper_triangle = df.where(np.triu(np.ones(df.shape), k=1).astype(bool), other="")

    return df_upper_triangle

# Modify the color_gradient function
def modified_color_gradient(val, min_val, max_val):
    if pd.isna(val) or val == "":
        return ''  # Return an empty string for NaN values or empty strings
    
    # Calculate the ratio
    ratio = (val - min_val) / (max_val - min_val)
    
    # Convert ratio to an RGB value (from yellow to light red)
    red = 255
    green = int(255 - (76 * ratio))  # difference between 255 and 179 is 76
    
    # Return the CSS string for the color
    return f'background-color: rgb({red}, {green}, 186)'

# Define a format function
def format_func(x):
    if isinstance(x, (float, int)):
        return "{:.3f}".format(x).rstrip('0').rstrip('.')
    return x

# Provided values
f = 170e9  # 170 GHz in Hz
a = 17.64e-3  # 17.64 mm in meters
m_values = list(range(0, 5))
n_values = list(range(1, 5))

# Generate the modified DataFrame
modified_df = modified_generate_wavelength_dataframe(f, a, m_values, n_values)

# Filter out empty strings and determine the min and max values
min_value = modified_df.replace("", np.nan).min().min()
max_value = modified_df.replace("", np.nan).max().max()

# Apply the modified color gradient with the filtered min and max values, and format the numbers
styled_modified_df = modified_df.style.applymap(
    lambda val: modified_color_gradient(val, min_value, max_value)
).format(format_func).set_caption("Beat Wavelengths (in meters)")

styled_modified_df.to_html("beat_wavelengths_table.html")