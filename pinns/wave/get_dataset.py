import numpy as np
import jax

from chart_autoencoder.get_charts import load_charts


def initial_conditions_spike(x, y, x0=1.0, y0=1.0, sigma=1.0, amplitude=10.0):
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


def initial_conditions_zero(x, y):
    return 0.0 * x


def get_dataset(charts_path, sigma=0.1, amplitude=50.0):

    (
        loaded_charts3d,
        loaded_charts_idxs,
        loaded_boundaries,
        loaded_boundary_indices,
        loaded_charts2d,
    ) = load_charts(charts_path)

    x = {}
    y = {}

    for chart_key in loaded_charts2d.keys():
        x[chart_key] = loaded_charts2d[chart_key][:, 0]
        y[chart_key] = loaded_charts2d[chart_key][:, 1]

    boundaries_x = {}
    boundaries_y = {}

    for key in loaded_boundary_indices.keys():
        start_boundary_indices = np.array(loaded_boundary_indices[key])

        starting_chart = key[0]
        starting_chart_points = loaded_charts2d[starting_chart][start_boundary_indices]
        ending_chart = key[1]

        if starting_chart not in boundaries_x:
            boundaries_x[starting_chart] = {}
        if starting_chart not in boundaries_y:
            boundaries_y[starting_chart] = {}

        boundaries_x[starting_chart][ending_chart] = starting_chart_points[:, 0]
        boundaries_y[starting_chart][ending_chart] = starting_chart_points[:, 1]


    # very simple initial conditions

    ics = {}
    ics[0] = initial_conditions_spike(x[0], y[0], sigma=sigma, amplitude=amplitude)
    for i in range(1, len(x)):
        ics[i] = initial_conditions_zero(x[i], y[i])
    
    ics_derivative = {}
    ics_derivative[0] = initial_conditions_zero(x[0], y[0])
    for i in range(1, len(x)):
        ics_derivative[i] = initial_conditions_zero(x[i], y[i])

    return x, y, ics, ics_derivative, boundaries_x, boundaries_y, loaded_charts3d