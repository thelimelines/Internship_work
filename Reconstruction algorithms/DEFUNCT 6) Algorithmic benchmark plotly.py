import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Output, Input
from dash.exceptions import PreventUpdate
from scipy.optimize import curve_fit, dual_annealing

# Function Definitions
# --------------------
def even_symmetric_fourier_series(x, *coeffs):
    return coeffs[0] + sum(coeff * np.cos(2 * n * np.pi * x / 180) for n, coeff in enumerate(coeffs[1:], 1))

def simulate_signal_with_polarization_shift(x_values, fourier_coefficients, weights, polarization_shifts):
    return sum(weight * even_symmetric_fourier_series((x_values + shift) % 180, *fourier_coefficients[(m, n)])
               for (m, n), weight, shift in zip(fourier_coefficients.keys(), weights, polarization_shifts))

def objective_scalar(params, x, y, fourier_coefficients):
    n_modes = len(fourier_coefficients)
    return np.sum((y - simulate_signal_with_polarization_shift(x, fourier_coefficients, params[:n_modes], params[n_modes:])) ** 2)

def plot_signals(x_values, original_signal, original_components, recovered_signal, recovered_components, sampled_x=None, sampled_y=None):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Signal and Components", "Recovered Signal and Components"))
    
    # Original Signal and Components
    fig.add_trace(go.Scatter(x=x_values, y=original_signal, mode='lines', name='Original Signal'), row=1, col=1)
    for component, label in original_components:
        fig.add_trace(go.Scatter(x=x_values, y=component, mode='lines', name=label), row=1, col=1)

    # Recovered Signal and Components
    fig.add_trace(go.Scatter(x=x_values, y=original_signal, mode='lines', name='Original Signal'), row=1, col=2)
    if sampled_x is not None and sampled_y is not None:
        fig.add_trace(go.Scatter(x=sampled_x, y=sampled_y, mode='markers', name='Sampled Points'), row=1, col=2)
    for component, label in recovered_components:
        fig.add_trace(go.Scatter(x=x_values, y=component, mode='lines', name=label), row=1, col=2)
        
    fig.add_trace(go.Scatter(x=x_values, y=recovered_signal, mode='lines', name='Recovered Signal'), row=1, col=2)
    
    fig.update_layout(height=600, width=1200, title_text="Original and Recovered Signals")
    return fig

# Data Loading and Preprocessing
# ------------------------------
data = pd.read_csv('Reconstruction algorithms\mode_data_test.csv')
unique_modes = data[['Mode m', 'mode n']].drop_duplicates().values

even_symmetric_fourier_coefficients_corrected = {}
for mode in unique_modes:
    m, n = mode
    mode_data = data[(data['Mode m'] == m) & (data['mode n'] == n)]
    n_terms_adjusted = m + 1
    mode_data = mode_data[mode_data['Polarization'] <= 90]
    x_data = mode_data['Polarization'].values
    y_data = np.nanmean([mode_data['Power1'].values, mode_data['Power2'].values], axis=0)
    popt, _ = curve_fit(even_symmetric_fourier_series, x_data, y_data, p0=[1.0] + [0.0] * n_terms_adjusted)
    even_symmetric_fourier_coefficients_corrected[(m, n)] = popt

# Dash App Initialization
# -----------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='error-graph', clickData=None),  # New graph for % error vs sample number
    dcc.Graph(id='main-graph'),
    dcc.Graph(id='trial-graph'),
    dcc.Dropdown(id='trial-dropdown', options=[{'label': f'Trial {i}', 'value': i} for i in range(10)], value=0),
    dcc.Store(id='trial-data-store'),
    dcc.Store(id='error-data-store')  # New dcc.Store for holding % error data
])

# Callbacks
# ---------
@app.callback(
    Output('error-graph', 'figure'),
    Input('error-data-store', 'data')  # The stored % error data
)
def update_error_graph(stored_data):
    if stored_data is None:
        raise PreventUpdate
    
    # Populate the error graph based on stored_data
    # ... (Your code to generate the figure here)
    # For example, you could do:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stored_data['Sample Number'], y=stored_data['Percent Error'], mode='markers'))
    fig.add_trace(go.Scatter(x=stored_data['Sample Number'], y=[np.mean(stored_data['Percent Error'])]*len(stored_data['Sample Number']), mode='lines'))
    return fig

@app.callback(
    Output('main-graph', 'figure'),
    [Input('error-graph', 'clickData'),  # Triggered when a point on error-graph is clicked
     Input('trial-data-store', 'data')]  # The stored trial data
)
def update_main_graph(clickData, stored_data):
    if clickData is None or stored_data is None:
        raise PreventUpdate
    
    point_index = clickData['points'][0]['pointIndex']
    
    if point_index in stored_data:
        trial_info = stored_data[point_index]
        return plot_signals(trial_info['x_values'], 
                            trial_info['original_signal'], 
                            trial_info['original_components'], 
                            trial_info['recovered_signal'], 
                            trial_info['recovered_components'],
                            trial_info['sampled_x'],
                            trial_info['sampled_y'])
    else:
        return go.Figure()

@app.callback(Output('trial-data-store', 'data'), Input('trial-dropdown', 'value'))

def update_store(selected_value):
    return trial_data

@app.callback(Output('trial-graph', 'figure'), [Input('main-graph', 'clickData'), Input('trial-data-store', 'data')])

def update_trial_graph(clickData, stored_data):
    if clickData is None or stored_data is None:
        return go.Figure()
    
    point_index = clickData['points'][0]['pointIndex']
    
    if point_index in stored_data:
        trial_info = stored_data[point_index]
        return plot_signals(trial_info['x_values'], 
                            trial_info['original_signal'], 
                            trial_info['original_components'], 
                            trial_info['recovered_signal'], 
                            trial_info['recovered_components'],
                            trial_info['sampled_x'],
                            trial_info['sampled_y'])
    else:
        return go.Figure()

# Data Simulation and Storage
# ---------------------------
avg_percent_diffs = []
all_percent_diffs = []
trial_data = {}

# Initialize result storage
avg_percent_diffs = []
all_percent_diffs = []
trial_count = 0 # Counter variable
trials = 1 # Number of trials per sample

# Loop over number of sample points
for n_points in range(3, 13):
    percent_diffs_for_this_n = []
    for trial in range(trials):
        # Generate random weights and shifts
        original_weights = np.round(np.random.uniform(0, 100, size=len(unique_modes)), 2)
        original_shifts = np.round(np.random.uniform(0, 180, size=len(unique_modes)), 2)
        
        # Generate the original synthetic signal
        x_values = np.linspace(0, 180, 1000)
        original_signal_values = simulate_signal_with_polarization_shift(x_values, even_symmetric_fourier_coefficients_corrected, original_weights, original_shifts)
        
        # Generate the original components for plotting
        original_components = []
        for (m, n), weight, shift in zip(even_symmetric_fourier_coefficients_corrected.keys(), original_weights, original_shifts):
            coeffs = even_symmetric_fourier_coefficients_corrected[(m, n)]
            shifted_x = (x_values + shift) % 180
            individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
            original_components.append((individual_mode_signal, f"{weight}x LP{m}{n} at {shift} deg shift"))

        # Sample the original signal
        sample_indices = np.linspace(0, len(x_values) - 1, n_points, dtype=int)
        sampled_x = x_values[sample_indices]
        sampled_y = original_signal_values[sample_indices]
        
        # Optimization
        bounds = [(0, max(original_weights))] * len(unique_modes) + [(0, 180)] * len(unique_modes)
        result = dual_annealing(objective_scalar, bounds, args=(sampled_x, sampled_y, even_symmetric_fourier_coefficients_corrected), maxiter=100, initial_temp=5230)
        recovered_params = result.x
        recovered_weights = recovered_params[:len(unique_modes)]
        recovered_shifts = recovered_params[len(unique_modes):]
        
        # Generate the recovered signal and its components
        recovered_signal_values = simulate_signal_with_polarization_shift(x_values, even_symmetric_fourier_coefficients_corrected, recovered_weights, recovered_shifts)
        recovered_components = []
        for (m, n), weight, shift in zip(even_symmetric_fourier_coefficients_corrected.keys(), recovered_weights, recovered_shifts):
            coeffs = even_symmetric_fourier_coefficients_corrected[(m, n)]
            shifted_x = (x_values + shift) % 180
            individual_mode_signal = weight * even_symmetric_fourier_series(shifted_x, *coeffs)
            recovered_components.append((individual_mode_signal, f"{weight:.2f}x LP{m}{n} at {shift:.2f} deg shift"))

        # Store trial data for later plotting
        trial_data[trial_count] = {
            'x_values': x_values,
            'original_signal': original_signal_values,
            'original_components': original_components,
            'recovered_signal': recovered_signal_values,
            'recovered_components': recovered_components,
            'sampled_x': sampled_x,
            'sampled_y': sampled_y
        }
        trial_count += 1

        # Compute percentage differences
        weight_percent_diffs = np.abs((recovered_weights - original_weights) / original_weights) * 100
        shift_percent_diffs = np.abs((recovered_shifts - original_shifts))  # Difference in degrees
        shift_percent_diffs = np.where(shift_percent_diffs > 90, 180 - shift_percent_diffs, shift_percent_diffs)  # Map to [0, 90]
        shift_percent_diffs = (shift_percent_diffs / 90) * 100  # Convert to percentage
        
        # Calculate total average percentage difference for this trial
        total_percent_diff = np.mean(np.concatenate([weight_percent_diffs, shift_percent_diffs]))
        percent_diffs_for_this_n.append(total_percent_diff)
    
    # Calculate and store the average percentage difference for this n_points
    avg_percent_diff = np.mean(percent_diffs_for_this_n)
    avg_percent_diffs.append(avg_percent_diff)
    all_percent_diffs.extend(percent_diffs_for_this_n)

error_data = {'Sample Number': list(range(len(avg_percent_diffs))), 'Percent Error': avg_percent_diffs}
@app.callback(
    Output('error-data-store', 'data'),
    Input('trial-dropdown', 'value')  # You can change this to your actual triggering condition
)
def update_error_store(selected_value):
    return error_data

# Run App
# -------
if __name__ == '__main__':
    app.run_server(debug=True)