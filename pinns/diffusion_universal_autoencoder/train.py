import os
import time
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

import json


import jax
import jax.numpy as jnp
import ml_collections
import models
from tqdm import tqdm
from jax.tree_util import tree_map
from samplers import (
    UniformICSampler,
    UniformSampler,
)

from chart_autoencoder import (
    get_metric_tensor_and_sqrt_det_g_grid_universal_autodecoder,
    load_charts3d,
)

from pinns.diffusion_universal_autoencoder.get_dataset import get_dataset

from pinns.diffusion_universal_autoencoder.plot import (
    plot_domains,
    plot_domains_3d,
    plot_domains_with_metric,
    plot_combined_3d_with_metric,
    plot_u0,
)

import wandb
from jaxpi.utils import save_checkpoint, load_config

from utils import set_profiler

import matplotlib.pyplot as plt
import numpy as np


def train_and_evaluate(config: ml_collections.ConfigDict):

    wandb_config = config.wandb
    run = wandb.init(
        project=wandb_config.project,
        name=wandb_config.name,
        entity="ricvalp",
        config=config,
    )

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    checkpoint_dir = f"{config.saving.checkpoint_dir}/{run.id}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir + "/cfg.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    X = np.linspace(0, 1, 50)
    Y = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(X, Y)
    coords = np.zeros((XX.size, 3))
    coords[:, 0] = XX.flatten()
    coords[:, 1] = YY.flatten()

    boundaries_x = np.zeros((XX.size, 3))
    boundaries_x[:, 0] = XX.flatten()
    boundaries_x[:, 1] = YY.flatten()

    boundaries_y = np.zeros((XX.size, 3))
    boundaries_y[:, 0] = XX.flatten()
    boundaries_y[:, 1] = YY.flatten()
    
    charts3d = []
    ts = np.linspace(0.0, 0.8, 200)
    for t in ts:
        charts3d.append(get_deformed_points(coords, t))

    plot_charts_sequence(charts3d, ts, name=Path(config.figure_path) / "charts_sequence.png")
    
    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), (conditionings, d_params) = get_metric_tensor_and_sqrt_det_g_grid_universal_autodecoder(
        autoencoder_cfg=autoencoder_config,
        cfg=config,
        charts=charts3d,
        coords=coords[:, :2],
        inverse=True,
    )

    x = coords[:, 0]
    y = coords[:, 1]

    def initial_conditions_spike(x, y, x0=0.5, y0=0.5, sigma=.1, amplitude=30.0):
        return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)

    u0 = initial_conditions_spike(x, y)


    plot_u0(x, y, u0, name=Path(config.figure_path) / "u0.png")

    plot_domains_with_metric(x, y, sqrt_det_g, conditionings, name="sequence.png")

    ics_sampler = iter(
        UniformICSampler(
            x=x,
            y=y,
            u0=u0,
            batch_size=config.training.batch_size,
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            ts=ts,
            sigma=0.01,
            batch_size=config.training.batch_size,
        )
    )

    model = models.DiffusionTime(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        conditionings=conditionings,
        ts=ts,
        ics=(x, y, u0),
    )

    print("Waiting for JIT...")

    for step in tqdm(range(1, config.training.max_steps + 1), desc="Training"):

        # set_profiler(config.profiler, step, config.profiler.log_dir)

        batch = next(res_sampler), next(ics_sampler)
        loss, model.state = model.step(model.state, batch)

        if step % config.wandb.log_every_steps == 0:
            wandb.log({"loss": loss}, step)
        
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                save_checkpoint(
                    model.state,
                    config.saving.checkpoint_dir,
                    keep=config.saving.num_keep_ckpts,
                )

    return model



def get_deformed_points(grid, t):
    """
    Apply a smooth deformation in the z-direction to the chart points.
    
    Args:
        chart_id: index of the chart to transform
        t: controls the strength of deformation (0.0 = no deformation)
    
    Returns:
        Transformed points
    """
    # Get the base points for this chart
    points = grid.copy()  # Make a copy to avoid modifying original data
    
    # Get x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Generate random frequencies (but still zero at boundaries)
    freq_x = np.array([1, 2])  # Random integer between 1 and 3
    freq_y = np.array([1, 2])
    
    # Create a 2D sine wave deformation in z-direction that is zero at the boundaries
    # Sum over various frequencies for a more complex deformation pattern
    deformation_z = np.zeros_like(x)
    for i in range(len(freq_x)):
        # Each term is zero when x=0, x=1, y=0, or y=1
        deformation_z += t * np.sin(freq_x[i] * np.pi * x) * np.sin(freq_y[i] * np.pi * y)
    
    # Apply the deformation to z coordinate
    points[:, 2] = deformation_z
    
    return points


def plot_charts_sequence(charts, ts, name=None):
    num_charts = len(charts)
    cols = min(5, num_charts)
    rows = (num_charts + cols - 1) // cols
    
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    
    for i, chart in enumerate(charts[::10]):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"t={ts[i]:.2f}")
        ax.scatter(
            chart[:, 0],
            chart[:, 1],
            chart[:, 2],
            s=3
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(-1, 1)
    
    plt.tight_layout()
    
    if name is not None:
        plt.savefig(name)
    plt.show()
    plt.close()


def plot_charts_sequence_with_solution(charts, ts, values=None, figsize=(15, 10), dpi=300, 
                         cmap='viridis', alpha=0.8, s=5, view_angle=(30, 45),
                         save_path=None, every_n=10, zlim=None, 
                         show_colorbar=False, show_time=True, vmin=None, vmax=None,
                         max_time_fraction=1.0):
    """
    Plot a sequence of 3D charts over time with publication-quality styling.
    
    Parameters:
    -----------
    charts : list of numpy.ndarray
        List of point clouds at different time steps
    ts : numpy.ndarray
        Time values corresponding to each chart
    values : list of numpy.ndarray, optional
        Values to color the points by at each time step (if None, uses z-coordinate)
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution in dots per inch
    cmap : str, optional
        Colormap for the points (default: 'viridis')
    alpha : float, optional
        Transparency of points (0 to 1)
    s : float, optional
        Point size
    view_angle : tuple, optional
        Elevation and azimuth angles for the 3D view
    save_path : str, optional
        Path to save the figure (if None, figure is not saved)
    every_n : int, optional
        Plot every n-th chart to reduce the number of subplots
    zlim : tuple, optional
        Limits for z-axis (min, max)
    show_colorbar : bool, optional
        Whether to show the colorbar
    show_time : bool, optional
        Whether to show time labels in subplot titles
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    max_time_fraction : float, optional
        Fraction of the time range to plot (default: 1.0 = full range)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of mpl_toolkits.mplot3d.Axes3D
        The 3D axes objects
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False
    
    # Limit to the specified time fraction
    max_time_idx = int(len(ts) * max_time_fraction)
    charts = charts[:max_time_idx]
    ts = ts[:max_time_idx]
    if values is not None:
        values = values[:max_time_idx]
    
    # Force exactly 5 plots per row
    cols = 5
    
    # Calculate how many plots to show (must be a multiple of 5)
    num_plots = (len(charts) // cols) * cols
    if num_plots > 10:  # Limit to 10 plots (2 rows)
        num_plots = 10
    
    rows = num_plots // cols
    
    # Calculate indices to select evenly spaced plots
    if len(charts) > num_plots:
        indices = np.linspace(0, len(charts) - 1, num_plots, dtype=int)
        selected_charts = [charts[i] for i in indices]
        selected_ts = ts[indices]
        if values is not None:
            selected_values = [values[i] for i in indices]
    else:
        # If we have fewer charts than needed, use all of them
        selected_charts = charts[:num_plots]
        selected_ts = ts[:num_plots]
        if values is not None:
            selected_values = values[:num_plots]
    
    # Create figure without using tight_layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Find global min/max for consistent color scaling and z-limits
    if zlim is None:
        all_z = np.concatenate([chart[:, 2] for chart in selected_charts])
        zlim = (np.min(all_z), np.max(all_z))
    
    if values is not None and vmin is None and vmax is None:
        all_values = np.concatenate(selected_values)
        vmin, vmax = np.min(all_values), np.max(all_values)
    
    # Calculate custom grid positions for maximum plot size
    # Define even smaller margins
    left_margin = 0.01
    right_margin = 0.01 if not show_colorbar else 0.10
    bottom_margin = 0.01
    top_margin = 0.01
    
    # Calculate available space
    available_width = 1.0 - left_margin - right_margin
    available_height = 1.0 - bottom_margin - top_margin
    
    # Calculate plot size and spacing - make plots overlap slightly
    plot_width = available_width / cols * 1.02  # Slightly larger than allocated space
    plot_height = available_height / rows * 1.02
    
    # Create subplots
    axes = []
    for i in range(num_plots):
        row = i // cols
        col = i % cols
        
        # Calculate position for this subplot - allow slight overlap
        left = left_margin + col * (available_width / cols) - 0.005
        bottom = 1.0 - top_margin - (row + 1) * (available_height / rows) - 0.005
        width = plot_width + 0.01  # Slightly larger to create overlap
        height = plot_height + 0.01
        
        # Create custom positioned subplot
        ax = fig.add_axes([left, bottom, width, height], projection="3d")
        axes.append(ax)
        
        chart = selected_charts[i]
        t = selected_ts[i]
        
        # Extract coordinates
        x, y, z = chart[:, 0], chart[:, 1], chart[:, 2]
        
        # Color by values if provided, otherwise by z-coordinate
        if values is not None:
            scatter = ax.scatter(
                x, y, z,
                c=selected_values[i],
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=vmin,
                vmax=vmax
            )
        else:
            scatter = ax.scatter(
                x, y, z,
                c=z,
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=zlim[0] if vmin is None else vmin,
                vmax=zlim[1] if vmax is None else vmax
            )
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set title with time - larger font size and minimal padding
        if show_time:
            ax.set_title(f"t = {t:.2f}", fontsize=18, pad=2)
        
        # Set consistent z-limits
        ax.set_zlim(zlim)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Make the panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make the panes (box faces) completely invisible
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Turn off the grid
        ax.grid(False)
        
        # Hide all axes
        ax.set_axis_off()
    
    # Add a single colorbar for all subplots if requested
    if show_colorbar:
        cbar_ax = fig.add_axes([1.0 - right_margin + 0.01, bottom_margin, 0.02, available_height])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        if values is not None:
            cbar.set_label('Value', fontsize=14, rotation=270, labelpad=20)
        else:
            cbar.set_label('Z Coordinate', fontsize=14, rotation=270, labelpad=20)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        # Also save as PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    return fig, axes


def plot_charts_individually(charts, ts, values=None, figsize=(8, 8), dpi=300, 
                         cmap='viridis', alpha=0.8, s=5, view_angle=(30, 45),
                         save_dir=None, zlim=None, vmin=None, vmax=None,
                         show_time=True, file_prefix="chart_t_", every_n=1):
    """
    Plot each 3D chart individually and save to separate files.
    
    Parameters:
    -----------
    charts : list of numpy.ndarray
        List of point clouds at different time steps
    ts : numpy.ndarray
        Time values corresponding to each chart
    values : list of numpy.ndarray, optional
        Values to color the points by at each time step (if None, uses z-coordinate)
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution in dots per inch
    cmap : str, optional
        Colormap for the points (default: 'viridis')
    alpha : float, optional
        Transparency of points (0 to 1)
    s : float, optional
        Point size
    view_angle : tuple, optional
        Elevation and azimuth angles for the 3D view
    save_dir : str, optional
        Directory to save the figures (if None, figures are not saved)
    zlim : tuple, optional
        Limits for z-axis (min, max)
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    show_time : bool, optional
        Whether to show time labels in subplot titles
    file_prefix : str, optional
        Prefix for saved files
    every_n : int, optional
        Save only one chart every N steps (default: 1 = save all charts)
    
    Returns:
    --------
    None
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False
    
    # Create save directory if it doesn't exist
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Select charts at regular intervals
    selected_indices = range(0, len(charts), every_n)
    selected_charts = [charts[i] for i in selected_indices]
    selected_ts = ts[selected_indices]
    if values is not None:
        selected_values = [values[i] for i in selected_indices]
    
    # Find global min/max for consistent color scaling and z-limits
    if zlim is None:
        all_z = np.concatenate([chart[:, 2] for chart in selected_charts])
        zlim = (np.min(all_z), np.max(all_z))
    
    if values is not None and vmin is None and vmax is None:
        all_values = np.concatenate(selected_values)
        vmin, vmax = np.min(all_values), np.max(all_values)
    
    # Process each selected chart individually
    for i, (chart, t) in enumerate(zip(selected_charts, selected_ts)):
        # Create a new figure for each chart
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        
        # Extract coordinates
        x, y, z = chart[:, 0], chart[:, 1], chart[:, 2]
        
        # Color by values if provided, otherwise by z-coordinate
        if values is not None:
            scatter = ax.scatter(
                x, y, z,
                c=selected_values[i],
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=vmin,
                vmax=vmax
            )
        else:
            scatter = ax.scatter(
                x, y, z,
                c=z,
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=zlim[0] if vmin is None else vmin,
                vmax=zlim[1] if vmax is None else vmax
            )
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set title with time
        if show_time:
            ax.set_title(f"t = {t:.2f}", fontsize=18)
        
        # Set consistent z-limits
        ax.set_zlim(zlim)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Make the panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make the panes (box faces) completely invisible
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Turn off the grid
        ax.grid(False)
        
        # Hide all axes
        ax.set_axis_off()
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        if values is not None:
            cbar.set_label('Value', fontsize=14, rotation=270, labelpad=20)
        else:
            cbar.set_label('Z Coordinate', fontsize=14, rotation=270, labelpad=20)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure if a directory is provided
        if save_dir:
            # Format the time value for the filename
            time_str = f"{t:.3f}".replace('.', '_')
            filename = f"{file_prefix}{time_str}.png"
            filepath = Path(save_dir) / filename
            plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
            
            # Also save as PDF
            pdf_path = str(filepath).rsplit('.', 1)[0] + '.pdf'
            plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        
        # Close the figure to free memory
        plt.close(fig)

