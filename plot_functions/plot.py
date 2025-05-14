import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.colors import Normalize


def plot_2d_scatter(points, figsize=(10, 8), dpi=300, cmap='viridis', 
                   alpha=0.8, s=30, color_by=None, save_path=None, 
                   title=None, show_colorbar=True, axis_labels=('X', 'Y'), 
                   grid=True, remove_ticks=True, xlim=None, ylim=None):
    """
    Plot a 2D scatter plot with high quality for publication.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Point data with shape (n_points, 2)
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
    color_by : numpy.ndarray, optional
        Values to color the points by (if None, uses the y-coordinate)
    save_path : str, optional
        Path to save the figure (if None, figure is not saved)
    title : str, optional
        Title of the plot
    show_colorbar : bool, optional
        Whether to show the colorbar
    axis_labels : tuple, optional
        Labels for the x and y axes
    grid : bool, optional
        Whether to show grid lines
    remove_ticks : bool, optional
        Whether to remove axis ticks
    xlim : tuple, optional
        Limits for x-axis (min, max)
    ylim : tuple, optional
        Limits for y-axis (min, max)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Extract x, y coordinates
    x, y = points[:, 0], points[:, 1]
    
    # Determine coloring
    if color_by is None:
        color_by = y
        colorbar_label = axis_labels[1]
    else:
        colorbar_label = 'Value'
    
    # Normalize color values
    norm = Normalize(vmin=np.min(color_by), vmax=np.max(color_by))
    
    # Plot the points
    scatter = ax.scatter(x, y, c=color_by, cmap=cmap, s=s, alpha=alpha, 
                         norm=norm, edgecolors='none')
    
    # Set axis labels with increased font size
    ax.set_xlabel(axis_labels[0], fontsize=14, labelpad=10)
    ax.set_ylabel(axis_labels[1], fontsize=14, labelpad=10)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
    
    # Remove axis ticks if requested
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    
    # Add colorbar if requested
    if show_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1, aspect=20)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(colorbar_label, fontsize=14, rotation=270, labelpad=20)
    
    # Set grid
    ax.grid(grid, alpha=0.3)
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('0.8')
        spine.set_linewidth(0.8)
    
    # Set tight layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig, ax



def plot_3d_pointcloud(points, figsize=(10, 8), dpi=300, cmap='viridis', 
                       alpha=0.8, s=10, view_angle=(30, 45), 
                       save_path=None, title=None, show_colorbar=True,
                       axis_labels=('X', 'Y', 'Z'), grid=True):
    """
    Plot a 3D point cloud with high quality for publication.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Point cloud data with shape (n_points, 3)
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
    title : str, optional
        Title of the plot
    show_colorbar : bool, optional
        Whether to show the colorbar
    axis_labels : tuple, optional
        Labels for the x, y, and z axes
    grid : bool, optional
        Whether to show grid lines
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Normalize z values for coloring
    norm = plt.Normalize(z.min(), z.max())
    
    # Plot the points
    scatter = ax.scatter(x, y, z, c=z, cmap=cmap, s=s, alpha=alpha, 
                         norm=norm, edgecolors='none')
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Set axis labels with increased font size
    ax.set_xlabel(axis_labels[0], fontsize=14, labelpad=10)
    ax.set_ylabel(axis_labels[1], fontsize=14, labelpad=10)
    ax.set_zlabel(axis_labels[2], fontsize=14, labelpad=10)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    
    # Add colorbar if requested
    if show_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1, aspect=20)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Z Coordinate', fontsize=14, rotation=270, labelpad=20)
    
    # Set grid
    ax.grid(grid)
    
    # Make the panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make the grid lines lighter
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Adjust the 3D plot to make it more visually appealing
    ax.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.6)
    ax.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.6)
    ax.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.6)
    
    # Set tight layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig, ax



# Example usage:
if __name__ == "__main__":
    # Generate a sample point cloud
    n_points = 1000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0.5, 1.0, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    points = np.column_stack((x, y, z))
    
    # Plot the point cloud
    fig, ax = plot_3d_pointcloud(
        points, 
        title="3D Point Cloud Example",
        save_path="high_quality_pointcloud.png"
    )
    
    plt.show()
