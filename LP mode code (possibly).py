import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

# Constants
a = 1
phi = np.linspace(0, 2 * np.pi, 400)
r = np.linspace(0, a, 400)
R, Phi = np.meshgrid(r, phi)
X, Y = R * np.cos(Phi), R * np.sin(Phi)

# Adjusting the compute_Ey function to only return the magnitude
def compute_Ey_magnitude(m, n):
    X_mn = jn_zeros(m, n)[-1]
    return jn(m, X_mn * R / a) * np.cos(m * Phi)

# Creating an 8x4 grid for the plots
fig, axes = plt.subplots(4, 8, figsize=(20, 14))
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Adjusting the colorbar position to avoid overlap

for m in range(4):  # m from 0 to 3
    for n in range(1, 5):  # n from 1 to 4
        # Plotting Heatmap
        ax_heatmap = axes[m, 2*(n-1)]
        Ey_magnitude = compute_Ey_magnitude(m, n)
        c = ax_heatmap.pcolormesh(X, Y, Ey_magnitude, shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax_heatmap.set_title(f"LP_{m}{n} Heatmap")
        ax_heatmap.set_aspect('equal')
        ax_heatmap.axis('off')
        
        # Plotting Quiver
        ax_quiver = axes[m, 2*(n-1)+1]
        skip = 20
        U = np.zeros_like(Ey_magnitude)  # no x component
        V = Ey_magnitude / np.max(Ey_magnitude)  # Normalize the magnitude to [0,1]
        ax_quiver.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip], 
                         color='black', scale=10, width=0.005)
        ax_quiver.set_title(f"LP_{m}{n} Quiver")
        ax_quiver.set_aspect('equal')
        ax_quiver.axis('off')

fig.colorbar(c, cax=cbar_ax)
fig.suptitle("Ey Magnitude and Direction for All Odd Modes of LP_mn", fontsize=16)
#plt.tight_layout()
plt.show()