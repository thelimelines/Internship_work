import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to plot signals and their components
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
def on_pick(event,input_folder):
    ind = event.ind[0]+1  # Get the index of the clicked point

    # Load the trial data from the corresponding CSV file
    df_loaded = pd.read_csv(f"{input_folder}trial_{ind}.csv")
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
    with open(f"{input_folder}trial_{ind}_labels.txt", "r") as label_file:
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

input_folder="Reconstruction algorithms\Benchmark_data/"

# Load the summary data for the main plot from separate CSV files
avg_sse_errors = pd.read_csv(f"{input_folder}avg_sse_errors.csv")['avg_sse_errors'].values
all_sse_errors = pd.read_csv(f"{input_folder}all_sse_errors.csv")['all_sse_errors'].values

# Create the main interactive plot
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(np.repeat(np.arange(3, 13), len(all_sse_errors) // 10), all_sse_errors, c='blue', alpha=0.5, picker=True)
ax.plot(np.arange(3, 13), avg_sse_errors, c='red', marker='o')
ax.set_xlabel('Number of Sample Points')
ax.set_ylabel('Total Average Sum of Squared Errors (W^2)')
ax.set_title('Total Average Sum of Squared Errors vs Number of Sample Points')
ax.grid(True)

# Connect the pick event to the on_pick function and pass the input_folder as a parameter
fig.canvas.callbacks.connect('pick_event', lambda event: on_pick(event, input_folder))
plt.show()