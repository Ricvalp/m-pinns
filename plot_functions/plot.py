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
    plt.rcParams['text.usetex'] = True  # Enable LaTeX
    
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
    
    # Set axis labels with increased font size and LaTeX formatting
    ax.set_xlabel(r'$x$', fontsize=16, labelpad=10)
    ax.set_ylabel(r'$y$', fontsize=16, labelpad=10)
    
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
    
    # Keep only the bottom and left spines (standard x and y axes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Set tick colors to black
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
    # Set tight layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        # Also save as PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    return fig, ax


def plot_3d_pointcloud(points, values, figsize=(3, 3), dpi=300, cmap='plasma', 
                       alpha=0.8, s=100, view_angle=(30, 45), 
                       save_path=None, title=None, show_colorbar=False,
                       axis_labels=('X', 'Y', 'Z')):

    # Create figure with minimal margins
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    ax.scatter(x, y, z, c=values, cmap=cmap, s=s, alpha=alpha, edgecolors='none')
    
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
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
    
    # Adjust the plot to fill more of the figure
    # Set very tight axis limits with no padding
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    
    # Maximize the plot area
    ax.set_position([0, 0, 1, 1])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        # Also save as PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0, format='pdf')
    
    plt.close()
    
    return fig, ax


def plot_3d_charts(charts, figsize=(3, 3), dpi=300, cmap='viridis', 
                  alpha=0.8, s=100, view_angle=(30, 45), 
                  save_path=None, show_colorbar=False, colors=None):
    """
    Plot multiple 3D point clouds in the same axis with high quality for publication.
    
    Parameters:
    -----------
    charts : dict or numpy.ndarray
        If dict: Dictionary of point clouds {id: points_array}
        If array: Single point cloud with shape (n_points, 3)
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
    show_colorbar : bool, optional
        Whether to show the colorbar
    colors : array-like, optional
        Custom colors for points (only used for single chart)
    
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
    plt.rcParams['text.usetex'] = False
    
    # Create figure with minimal margins
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')
    
    # Check if we have a single chart with custom colors
    is_single_chart_with_colors = not isinstance(charts, dict) and colors is not None
    
    # Convert single array to dict format for consistent processing
    if not isinstance(charts, dict):
        charts = {0: charts}
    
    # Get colormap with enough colors for all charts
    cmap_obj = cm.get_cmap(cmap)
    chart_colors = [cmap_obj(i/max(1, len(charts)-1)) for i in range(len(charts))]
    
    # Track min/max coordinates for axis limits
    all_x, all_y, all_z = [], [], []
    
    # Plot each chart
    for i, (chart_id, points) in enumerate(charts.items()):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Store coordinates for setting axis limits later
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)
        
        # Use custom colors for single chart if provided
        if i == 0 and is_single_chart_with_colors:
            scatter = ax.scatter(x, y, z, c=colors, cmap=cmap, s=s, alpha=alpha, edgecolors='none')
            if show_colorbar:
                cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1, aspect=20)
                cbar.ax.tick_params(labelsize=12)
        else:
            # Plot the points with a unique color for each chart
            ax.scatter(x, y, z, color=chart_colors[i], s=s, alpha=alpha, edgecolors='none')
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
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
    
    # Set axis limits based on all points
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))
    
    # Maximize the plot area
    ax.set_position([0, 0, 1, 1])
    
    # Add colorbar for multiple charts if requested
    if show_colorbar and len(charts) > 1 and not is_single_chart_with_colors:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(0, len(charts)-1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, aspect=20)
        cbar.ax.tick_params(labelsize=12)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        # Also save as PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0, format='pdf')
    
    return fig, ax


def plot_3d_charts_solution(charts, figsize=(3, 3), dpi=300, cmap='plasma', title=None,
                  alpha=0.8, s=100, view_angle=(30, 45), 
                  save_path=None, show_colorbar=False, colors=None):
    """
    Plot multiple 3D point clouds in the same axis with high quality for publication.
    
    Parameters:
    -----------
    charts : dict or numpy.ndarray
        If dict: Dictionary of point clouds {id: points_array}
        If array: Single point cloud with shape (n_points, 3)
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
    show_colorbar : bool, optional
        Whether to show the colorbar
    colors : array-like, optional
        Custom colors for points (only used for single chart)
    
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
    plt.rcParams['text.usetex'] = False
    
    # Create figure with minimal margins
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Set extremely tight margins (left, bottom, right, top)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Create the 3D axes with position that fills the entire figure
    ax = fig.add_subplot(111, projection='3d', position=[0, 0, 1, 1])
    
    # Check if we have a single chart with custom colors
    is_single_chart_with_colors = not isinstance(charts, dict) and colors is not None
    
    # Convert single array to dict format for consistent processing
    if not isinstance(charts, dict):
        charts = {0: charts}
    
    # Get colormap with enough colors for all charts
    cmap_obj = cm.get_cmap(cmap)
    chart_colors = [cmap_obj(i/max(1, len(charts)-1)) for i in range(len(charts))]
    
    # Track min/max coordinates for axis limits
    all_x, all_y, all_z = [], [], []
    
    # Plot each chart
    for i, (chart_id, points) in enumerate(charts.items()):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Store coordinates for setting axis limits later
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)
        
        # Use custom colors for single chart if provided
        if i == 0 and is_single_chart_with_colors:
            scatter = ax.scatter(x, y, z, c=colors, cmap=cmap, s=s, alpha=alpha, edgecolors='none')
            if show_colorbar:
                cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05, aspect=20)
                cbar.ax.tick_params(labelsize=12)
        else:
            # Plot the points with a unique color for each chart
            ax.scatter(x, y, z, color=chart_colors[i], s=s, alpha=alpha, edgecolors='none')
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
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
    
    # Set axis limits based on all points
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    
    # Add colorbar for multiple charts if requested
    if show_colorbar and len(charts) > 1 and not is_single_chart_with_colors:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(0, len(charts)-1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.05, aspect=20)
        cbar.ax.tick_params(labelsize=12)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        # Also save as PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0, format='pdf')
    
    return fig, ax


def plot_domains(x, y, boundaries_x, boundaries_y, name=None, figsize=(15, 15), 
                dpi=300, cmap='viridis', alpha=0.8, s=3, boundary_s=10, 
                remove_ticks=True, show_legend=True):
    """
    Plot 2D scatter plots with boundaries in a grid layout with publication-quality style.
    
    Parameters:
    -----------
    x : list of arrays
        List of x-coordinates for each chart
    y : list of arrays
        List of y-coordinates for each chart
    boundaries_x : dict of dicts
        Dictionary where boundaries_x[m][n] contains x-coordinates of the boundary 
        between chart m and chart n (on chart m)
    boundaries_y : dict of dicts
        Dictionary where boundaries_y[m][n] contains y-coordinates of the boundary 
        between chart m and chart n (on chart m)
    name : str, optional
        Path to save the figure (if None, figure is not saved)
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution in dots per inch
    cmap : str, optional
        Colormap for the points (default: 'viridis')
    alpha : float, optional
        Transparency of points (0 to 1)
    s : float, optional
        Point size for main scatter points
    boundary_s : float, optional
        Point size for boundary points
    remove_ticks : bool, optional
        Whether to remove axis ticks
    show_legend : bool, optional
        Whether to show the legend
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : array of matplotlib.axes.Axes
        The axes objects
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False
    
    # Determine the number of plots needed
    num_plots = len(x)
    cols = min(4, num_plots)  # Adjust columns based on number of plots
    rows = (num_plots + cols - 1) // cols  # Calculate required rows
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    
    # Ensure axes is a 2D array for easy indexing
    if num_plots == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get colormap for boundaries
    boundary_cmap = plt.cm.get_cmap('tab10')
    
    # Plot each chart
    for i in range(num_plots):
        # Calculate row and column index for the plot
        row, col = divmod(i, cols)
        
        # Plot main scatter points
        axes[row, col].scatter(x[i], y[i], s=s, color='#1f77b4', alpha=alpha, edgecolors='none')
        
        # Plot boundaries for this chart if they exist
        if i in boundaries_x:
            # Track which boundaries we've already plotted to avoid duplicates in the legend
            plotted_boundaries = set()
            
            # Plot each boundary for this chart
            for j, neighbor_chart in enumerate(boundaries_x[i]):
                # Create a unique identifier for this boundary pair (smaller index first)
                boundary_id = f"{min(i, neighbor_chart)}-{max(i, neighbor_chart)}"
                
                # Only add to legend if we haven't seen this boundary before
                if boundary_id not in plotted_boundaries:
                    axes[row, col].scatter(
                        boundaries_x[i][neighbor_chart], 
                        boundaries_y[i][neighbor_chart], 
                        s=boundary_s, 
                        color=boundary_cmap(j % 10),
                        label=f"boundary {i}-{neighbor_chart}",
                        alpha=1.0,
                        edgecolors='none'
                    )
                    plotted_boundaries.add(boundary_id)
                else:
                    # Plot without adding to legend
                    axes[row, col].scatter(
                        boundaries_x[i][neighbor_chart], 
                        boundaries_y[i][neighbor_chart], 
                        s=boundary_s, 
                        color=boundary_cmap(j % 10),
                        alpha=1.0,
                        edgecolors='none'
                    )
        
        # Set title with minimal styling
        axes[row, col].set_title(f"Chart {i}", fontsize=12, pad=10)
        
        # Remove ticks if requested
        if remove_ticks:
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        
        # Add legend if requested
        if show_legend:
            axes[row, col].legend(loc='best', frameon=True, framealpha=0.7, fontsize=8)
        
        # Remove grid
        axes[row, col].grid(False)
        
        # Set background color to white
        axes[row, col].set_facecolor('white')
    
    # Hide unused subplots
    for i in range(num_plots, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if name is not None:
        plt.savefig(name, bbox_inches='tight', dpi=dpi)
        # Also save as PDF
        pdf_path = name.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    return fig, axes


def plot_domains_subset(x, y, boundaries_x, boundaries_y, indices=[0, 1, 2, 3], name=None, 
                       figsize=(15, 15), dpi=300, cmap='viridis', alpha=0.8, s=3, 
                       boundary_s=10, remove_ticks=True, show_legend=True):
    """
    Plot a subset of 2D scatter plots with boundaries in a grid layout with publication-quality style.
    
    Parameters:
    -----------
    x : list of arrays
        List of x-coordinates for each chart
    y : list of arrays
        List of y-coordinates for each chart
    boundaries_x : dict of dicts
        Dictionary where boundaries_x[m][n] contains x-coordinates of the boundary 
        between chart m and chart n (on chart m)
    boundaries_y : dict of dicts
        Dictionary where boundaries_y[m][n] contains y-coordinates of the boundary 
        between chart m and chart n (on chart m)
    indices : list, optional
        List of indices to plot (default: [0, 1, 2, 3])
    name : str, optional
        Path to save the figure (if None, figure is not saved)
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution in dots per inch
    cmap : str, optional
        Colormap for the points (default: 'viridis')
    alpha : float, optional
        Transparency of points (0 to 1)
    s : float, optional
        Point size for main scatter points
    boundary_s : float, optional
        Point size for boundary points
    remove_ticks : bool, optional
        Whether to remove axis ticks
    show_legend : bool, optional
        Whether to show the legend
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : array of matplotlib.axes.Axes
        The axes objects
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False
    
    # Determine the grid layout
    rows = 2
    cols = 2
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    
    # Get colormap for boundaries - changed from 'viridis' to 'magma'
    boundary_cmap = plt.cm.get_cmap('magma')
    
    # Plot each chart
    for idx, i in enumerate(indices):
        if idx >= rows * cols:
            break
            
        # Calculate row and column index for the plot
        row, col = divmod(idx, cols)
        
        # Plot main scatter points
        axes[row, col].scatter(x[i], y[i], s=s, color='#1f77b4', alpha=alpha, edgecolors='none')
        
        # Plot boundaries for this chart if they exist
        if i in boundaries_x:
            # Track which boundaries we've already plotted to avoid duplicates in the legend
            plotted_boundaries = set()
            
            # Get all neighbor charts for normalization
            all_neighbors = list(boundaries_x[i].keys())
            total_boundaries = len(all_neighbors)
            
            # Plot each boundary for this chart
            for j, neighbor_chart in enumerate(all_neighbors):
                # Create a unique identifier for this boundary pair (smaller index first)
                boundary_id = f"{min(i, neighbor_chart)}-{max(i, neighbor_chart)}"
                
                # Calculate normalized position in colormap (0 to 1)
                color_pos = j / max(1, total_boundaries - 1) if total_boundaries > 1 else 0.5
                
                # Only add to legend if we haven't seen this boundary before
                if boundary_id not in plotted_boundaries:
                    axes[row, col].scatter(
                        boundaries_x[i][neighbor_chart], 
                        boundaries_y[i][neighbor_chart], 
                        s=boundary_s, 
                        color=boundary_cmap(color_pos),  # Use normalized position in colormap
                        label=f"boundary {i}-{neighbor_chart}",
                        alpha=1.0,
                        edgecolors='none'
                    )
                    plotted_boundaries.add(boundary_id)
                else:
                    # Plot without adding to legend
                    axes[row, col].scatter(
                        boundaries_x[i][neighbor_chart], 
                        boundaries_y[i][neighbor_chart], 
                        s=boundary_s, 
                        color=boundary_cmap(color_pos),  # Use normalized position in colormap
                        alpha=1.0,
                        edgecolors='none'
                    )
        
        # Set title with chart index
        axes[row, col].set_title(f"Chart {i}", fontsize=12, pad=10)
        
        # Remove ticks if requested
        if remove_ticks:
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        
        # Add legend if requested
        if show_legend:
            axes[row, col].legend(loc='best', frameon=True, framealpha=0.7, fontsize=8)
        
        # Remove grid
        axes[row, col].grid(False)
        
        # Set background color to white
        axes[row, col].set_facecolor('white')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if name is not None:
        plt.savefig(name, bbox_inches='tight', dpi=dpi)
        # Also save as PDF
        pdf_path = name.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    return fig, axes


# Example usage:
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import cm
    
    # Test plot_3d_charts with multiple point clouds
    # Generate two sample point clouds on different parts of a sphere
    n_points = 500
    
    # First point cloud (top part of sphere)
    phi1 = np.random.uniform(0, np.pi/3, n_points)
    theta1 = np.random.uniform(0, 2*np.pi, n_points)
    r1 = 1.0
    
    x1 = r1 * np.cos(theta1) * np.sin(phi1)
    y1 = r1 * np.sin(theta1) * np.sin(phi1)
    z1 = r1 * np.cos(phi1)
    
    points1 = np.column_stack((x1, y1, z1))
    
    # Second point cloud (bottom part of sphere)
    phi2 = np.random.uniform(2*np.pi/3, np.pi, n_points)
    theta2 = np.random.uniform(0, 2*np.pi, n_points)
    r2 = 1.0
    
    x2 = r2 * np.cos(theta2) * np.sin(phi2)
    y2 = r2 * np.sin(theta2) * np.sin(phi2)
    z2 = r2 * np.cos(phi2)
    
    points2 = np.column_stack((x2, y2, z2))
    
    # Create a dictionary of point clouds
    charts_dict = {0: points1, 1: points2}
    
    # Test with dictionary of point clouds
    fig, ax = plot_3d_charts(
        charts_dict,
        save_path="multiple_pointclouds.png",
        view_angle=(30, 45),
        alpha=0.8,
        cmap='cool',
        show_colorbar=False  # No colorbar
    )
    
    # Test with single point cloud and custom colors
    fig, ax = plot_3d_charts(
        points1,
        save_path="single_pointcloud_custom_colors.png",
        view_angle=(20, 110),
        alpha=1.0,
        colors=points1[:, 2],  # Color by z-coordinate
        cmap='plasma',
        show_colorbar=True
    )

    # Test plot_domains function
    # Generate sample data for multiple charts
    n_charts = 6
    n_points = 200
    chart_x = []
    chart_y = []
    
    # Create sample data for each chart
    for i in range(n_charts):
        # Generate random points in a circle with some noise
        theta = np.random.uniform(0, 2*np.pi, n_points)
        r = 0.8 * np.random.uniform(0.2, 1.0, n_points)
        
        # Add some offset to separate the charts visually
        offset_x = (i % 3) * 0.5
        offset_y = (i // 3) * 0.5
        
        x = r * np.cos(theta) + offset_x
        y = r * np.sin(theta) + offset_y
        
        chart_x.append(x)
        chart_y.append(y)
    
    # Create sample boundary data with meaningful labels
    boundaries_x = {}
    boundaries_y = {}
    
    # Create boundaries between specific chart pairs
    boundary_pairs = [(0, 1), (1, 2), (3, 4)]
    
    for i, (a, b) in enumerate(boundary_pairs):
        # Create a boundary between chart a and chart b
        theta = np.linspace(0, 2*np.pi, 50)
        r = 0.9 + i * 0.1  # Different radius for each boundary
        
        # Use the chart pair as the key
        key = f"{a} {b}"  # This will create labels like "boundary 0 1"
        
        boundaries_x[a] = {b: r * np.cos(theta)}
        boundaries_y[a] = {b: r * np.sin(theta)}
    
    # Test the plot_domains function
    fig, axes = plot_domains(
        chart_x, 
        chart_y, 
        boundaries_x, 
        boundaries_y, 
        name="domain_plots.png",
        figsize=(12, 10),
        s=5,
        boundary_s=15,
        remove_ticks=True,
        show_legend=True
    )
