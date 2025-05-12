import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
import pandas as pd


def plot_domains(x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, name=None):
    num_plots = len(x)
    cols = 4  # You can adjust the number of columns based on your preference
    rows = (num_plots + cols - 1) // cols  # Calculate required rows

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    # Ensure ax is a 2D array for easy indexing
    if num_plots == 1:
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    for i in range(num_plots):
        # Calculate row and column index for the plot
        row, col = divmod(i, cols)
        ax[row][col].set_title(f"Chart {i}")
        scatter = ax[row][col].scatter(x[i], y[i], s=3, c="b")
        scatter_bcs = ax[row][col].scatter(
            bcs_x[i], bcs_y[i], s=50, c=bcs[i], label="BCs"
        )
        # Add colorbar for boundary conditions
        if len(np.unique(bcs[i])) > 1:  # Only add colorbar if there are multiple colors
            fig.colorbar(
                scatter_bcs, ax=ax[row][col], orientation="vertical", label="BC Value"
            )

        # Plot boundaries for current chart
        if i in boundaries_x:
            for other_chart, boundary_x in boundaries_x[i].items():
                ax[row][col].scatter(
                    boundary_x,
                    boundaries_y[i][other_chart],
                    s=10,
                    label=f"boundary {i}-{other_chart}",
                )

        ax[row][col].legend(loc="best")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_with_metric(x, y, sqrt_det_g, d_params, name=None):
    num_plots = len(x)
    cols = 4
    rows = (num_plots + cols - 1) // cols

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_plots == 1:
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(num_plots):
        row, col = divmod(i, cols)

        ax[row][col].set_title(f"Chart {i}")
        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))
        scatter = ax[row][col].scatter(x[i], y[i], s=3, c=color_values, cmap="viridis")
        fig.colorbar(scatter, ax=ax[row][col], orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_3d(x, y, bcs_x, bcs_y, bcs, decoder, d_params, name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Combined 3D Plot")

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]

    # Create a single colorbar for all BC points
    all_bc_values = np.concatenate([bcs[i] for i in range(len(bcs))])
    vmin, vmax = np.min(all_bc_values), np.max(all_bc_values)

    for i in range(len(x)):
        p = decoder.apply({"params": decoder_params[i]}, np.stack([x[i], y[i]], axis=1))
        p_bcs = decoder.apply(
            {"params": decoder_params[i]}, np.stack([bcs_x[i], bcs_y[i]], axis=1)
        )

        # Plot the domain points
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=3, alpha=0.5, label=f"Chart {i}")

        # Plot the boundary conditions with colors
        scatter_bcs = ax.scatter(
            p_bcs[:, 0],
            p_bcs[:, 1],
            p_bcs[:, 2],
            c=bcs[i],
            s=50,
            vmin=vmin,
            vmax=vmax,
            label=f"BCs {i}",
        )

    # Add a single colorbar for all boundary conditions
    cbar = fig.colorbar(scatter_bcs, ax=ax, orientation="vertical", label="BC Value")

    # Set consistent axes limits
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.legend(loc="best")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_3d_with_metric(x, y, decoder, sqrt_det_g, d_params, name=None):
    num_plots = len(x)
    cols = 2
    rows = (num_plots + cols - 1) // cols

    fig = plt.figure(figsize=(15, 5 * rows))

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(num_plots):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"Chart {i}")
        points_3d = decoder.apply(
            {"params": decoder_params[i]}, jnp.stack([x[i], y[i]], axis=1)
        )
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))

        scatter = ax.scatter(x_3d, y_3d, z_3d, c=color_values, cmap="viridis", s=3)

        ax.legend(loc="best")
        fig.colorbar(scatter, ax=ax, orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_combined_3d_with_metric(x, y, decoder, sqrt_det_g, d_params, name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Combined 3D Plot with Metric Coloring")

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(len(x)):
        # Decode the 2D points to 3D using the decoder function
        points_3d = decoder.apply(
            {"params": decoder_params[i]}, jnp.stack([x[i], y[i]], axis=1)
        )
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        # Calculate the color values using sqrt_det_gs
        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))

        scatter = ax.scatter(
            x_3d, y_3d, z_3d, c=color_values, cmap="viridis", s=3, label=f"Chart {i}"
        )

    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_charts_solution(x, y, u_preds, name, vmin=None, vmax=None):

    if vmin is None:
        vmin = min(np.min(u_preds[key]) for key in u_preds.keys())
    if vmax is None:
        vmax = max(np.max(u_preds[key]) for key in u_preds.keys())

    num_charts = len(x)
    num_rows = int(np.ceil(np.sqrt(num_charts)))
    num_cols = int(np.ceil(num_charts / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 18))
    if num_charts == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, key in zip(axes, u_preds.keys()):
        ax.set_title(f"Chart {key}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        scatter = ax.scatter(
            x[key], y[key], c=u_preds[key], cmap="jet", s=100.0, vmin=vmin, vmax=vmax
        )
        fig.colorbar(scatter, ax=ax, shrink=0.6)

    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
    plt.close()


def plot_3d_level_curves(pts, sol, tol, angles=(30, 45), name=None):

    num_levels = 10
    levels = np.linspace(np.min(sol), np.max(sol), num_levels)

    colors = sol.copy()

    for level in levels:
        mask = np.abs(sol - level) < tol
        colors[mask] = np.nan

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        pts[~np.isnan(colors), 0],
        pts[~np.isnan(colors), 1],
        pts[~np.isnan(colors), 2],
        c=colors[~np.isnan(colors)],
        cmap="jet",
        s=1,
    )

    ax.scatter(
        pts[np.isnan(colors), 0],
        pts[np.isnan(colors), 1],
        pts[np.isnan(colors), 2],
        color="black",
        s=20,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("solution")

    ax.view_init(angles[0], angles[1])

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_3d_solution(pts, sol, angles, name=None, **kwargs):
    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    scatter = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=sol,
        cmap="jet",
        **kwargs,
    )

    ax.view_init(angles[0], angles[1])

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label("solution")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.close()


# def plot_correlation(mesh_sol, gt_sol, data=None, name=None):

#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.scatter(mesh_sol, gt_sol, s=5)
#     if data is not None:
#         ax.scatter(data, data, s=25, c="red")
#     ax.set_xlabel("Mesh Solution")
#     ax.set_ylabel("Ground Truth Solution")
#     ax.set_title("Correlation between Mesh and GT Solutions")

#     min_val = min(np.min(mesh_sol), np.min(gt_sol))
#     max_val = max(np.max(mesh_sol), np.max(gt_sol))
#     ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

#     if name is not None:
#         plt.savefig(name)
#     plt.show()

#     return fig


def plot_correlation(mesh_sol, gt_sol, data=None, name=None, min_val=0.0, max_val=40.0):
    """
    Generates a correlation plot using seaborn and matplotlib.

    Args:
        mesh_sol (np.ndarray): Array of mesh solution values.
        gt_sol (np.ndarray): Array of ground truth solution values.
        data (np.ndarray, optional): Array of data to plot as a reference line. Defaults to None.
        name (str, optional): Base name for the saved plot files. Defaults to "correlation_plot".

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    # Set a professional style using seaborn
    sns.set_theme(style="ticks")
    plt.rcParams["font.family"] = (
        "serif"  # Use a serif font for better readability in papers
    )
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12  # Increased tick label size
    plt.rcParams["ytick.labelsize"] = 12  # Increased tick label size
    plt.rcParams["legend.fontsize"] = 18  # Increased legend font size

    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size for better aspect ratio

    # Use different sizes for the plot but will standardize in the legend
    pinn_size = 30
    train_size = 70

    # Plot the main data points
    sns.scatterplot(
        x=mesh_sol,
        y=gt_sol,
        s=pinn_size,
        ax=ax,
        label=r"$\mathcal{M}$-PINN",
        edgecolor="none",
    )

    if data is not None:
        sns.scatterplot(
            x=data,
            y=data,
            s=train_size,
            color="red",
            marker="o",
            label="train points",
            ax=ax,
            edgecolor="none",
        )

    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    # ax.set_title("Correlation between Mesh and Ground Truth Solutions")

    min_val = 0.0
    if max_val is None:
        max_val = max(np.max(mesh_sol), np.max(gt_sol))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

    # Make legend markers the same size (standardize to a medium size)
    legend = ax.legend(prop={"size": 18})  # Increased legend size
    legend_marker_size = 50  # Increased legend marker size
    for handle in legend.legend_handles:
        handle._sizes = [legend_marker_size]

    # Fewer ticks with larger size
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce number of ticks on x-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce number of ticks on y-axis
    ax.tick_params(
        axis="both", which="major", labelsize=18, width=1.5, length=6
    )  # Bigger ticks

    # Enable grid for easier visual comparison
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.2)

    # Ensure tight layout to prevent labels from overlapping
    plt.tight_layout()

    if name is not None:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(
            f"{name}.png", format="png", dpi=300, bbox_inches="tight"
        )  # Higher DPI for better image quality

    plt.show()

    return fig


def plot_ablation(mpinn_csv, deltapinn_csv, log_scale=True, name=None):
    """
    Plots ablation study results from both MPINN and DeltaPINN CSV files, averaging over seeds.

    Args:
        mpinn_csv (str): Path to the CSV file containing MPINN results
        deltapinn_csv (str): Path to the CSV file containing DeltaPINN results
        name (str, optional): Base name for saving the plot files
    """
    # Set style consistent with plot_correlation
    sns.set_theme(style="ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 18

    # Read and process MPINN data
    mpinn_df = pd.read_csv(mpinn_csv)
    mpinn_grouped = (
        mpinn_df.groupby("N")
        .agg({"mpinn_corr": ["mean", "std"], "mpinn_mse": ["mean", "std"]})
        .reset_index()
    )

    # Read and process DeltaPINN data
    deltapinn_df = pd.read_csv(deltapinn_csv)
    deltapinn_grouped = (
        deltapinn_df.groupby("N")
        .agg({"deltapinn_corr": ["mean", "std"], "deltapinn_mse": ["mean", "std"]})
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot correlation
    ax1.errorbar(
        mpinn_grouped["N"],
        mpinn_grouped["mpinn_corr"]["mean"],
        yerr=mpinn_grouped["mpinn_corr"]["std"],
        marker="o",
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=2,
        color="blue",
        label=r"$\mathcal{M}$-PINN",
    )

    ax1.errorbar(
        deltapinn_grouped["N"],
        deltapinn_grouped["deltapinn_corr"]["mean"],
        yerr=deltapinn_grouped["deltapinn_corr"]["std"],
        marker="o",
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=2,
        color="red",
        label=r"$\Delta$-PINN",
    )

    ax1.set_xlabel("number of train points")
    ax1.set_ylabel("correlation")
    ax1.grid(True, linestyle="--", alpha=0.6, linewidth=1.2)

    # Plot MSE
    ax2.errorbar(
        mpinn_grouped["N"],
        mpinn_grouped["mpinn_mse"]["mean"],
        yerr=mpinn_grouped["mpinn_mse"]["std"],
        marker="o",
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=2,
        color="blue",
        label=r"$\mathcal{M}$-PINN",
    )

    ax2.errorbar(
        deltapinn_grouped["N"],
        deltapinn_grouped["deltapinn_mse"]["mean"],
        yerr=deltapinn_grouped["deltapinn_mse"]["std"],
        marker="o",
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=2,
        color="red",
        label=r"$\Delta$-PINN",
    )

    ax2.set_xlabel("number of train points")
    ax2.set_ylabel("MSE")
    ax2.grid(True, linestyle="--", alpha=0.6, linewidth=1.2)

    # Adjust ticks and appearance
    for ax in [ax1, ax2]:
        ax.tick_params(axis="both", which="major", labelsize=18, width=1.5, length=6)
        ax.legend(prop={"size": 18})
    if log_scale:
        ax2.set_yscale("log")

    plt.tight_layout()

    if name is not None:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{name}.png", format="png", dpi=300, bbox_inches="tight")

    plt.show()
    return fig


def plot_2d_scatter(points, colors=None, marker_size=50, colormap='viridis', alpha=0.8, 
                   name=None, figsize=(8, 6), colorbar_label=None, edge_color='black', 
                   edge_width=0.3, show_grid=True, grid_alpha=0.2, show_axes=True,
                   axes_linewidth=1.5, axes_color='black', x_label=None, y_label=None):
    """
    Creates a publication-quality 2D scatter plot with transparent background grid.
    
    Args:
        points (np.ndarray): Array of shape (n, 2) containing 2D point coordinates.
        colors (np.ndarray, optional): Array of values to color the points by. Defaults to None.
        marker_size (int or np.ndarray, optional): Size of markers. Can be an array for variable sizes. Defaults to 50.
        colormap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
        alpha (float, optional): Transparency of points (0 to 1). Defaults to 0.8.
        name (str, optional): Base name for saving the plot files. Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to (8, 6).
        colorbar_label (str, optional): Label for the colorbar. Defaults to None.
        edge_color (str, optional): Color of point edges. Defaults to 'black'.
        edge_width (float, optional): Width of point edges. Defaults to 0.3.
        show_grid (bool, optional): Whether to show the background grid. Defaults to True.
        grid_alpha (float, optional): Transparency of the grid. Defaults to 0.2.
        show_axes (bool, optional): Whether to show axes. Defaults to True.
        axes_linewidth (float, optional): Width of axes lines. Defaults to 1.5.
        axes_color (str, optional): Color of axes. Defaults to 'black'.
        x_label (str, optional): Label for x-axis. Defaults to None.
        y_label (str, optional): Label for y-axis. Defaults to None.
    
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    # Set a professional style using seaborn
    sns.set_theme(style="ticks")
    plt.rcParams["font.family"] = "serif"  # Use a serif font for better readability in papers
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate axis limits with a small buffer
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Add a small buffer (5% of range) to avoid points at the edges
    buffer_x = 0.05 * (x_max - x_min)
    buffer_y = 0.05 * (y_max - y_min)
    
    # Set axis limits
    ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
    ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
    
    # Main scatter plot
    if colors is not None:
        scatter = ax.scatter(
            points[:, 0], points[:, 1],
            c=colors, 
            cmap=colormap,
            s=marker_size,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=edge_width
        )
        
        # Add colorbar with professional styling
        if np.unique(colors).size > 1:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
            if colorbar_label:
                cbar.set_label(colorbar_label, size=18)
            cbar.ax.tick_params(labelsize=12)
    else:
        scatter = ax.scatter(
            points[:, 0], points[:, 1],
            s=marker_size,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=edge_width
        )
    
    # Configure axes
    if show_axes:
        # Set axis styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(axes_linewidth)
        ax.spines['left'].set_linewidth(axes_linewidth)
        ax.spines['bottom'].set_color(axes_color)
        ax.spines['left'].set_color(axes_color)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', width=1.5, length=6, pad=8, 
                      bottom=True, left=True, top=False, right=False)
        
        # Set axis labels if provided
        if x_label:
            ax.set_xlabel(x_label, labelpad=10)
        if y_label:
            ax.set_ylabel(y_label, labelpad=10)
    else:
        # Hide all spines and ticks if axes are not shown
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Configure grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=grid_alpha, linewidth=0.8)
    else:
        ax.grid(False)
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save high-quality versions if name is provided
    if name is not None:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{name}.png", format="png", dpi=600, bbox_inches="tight")
    
    plt.show()
    return fig


def plot_metrics_separately(correlation_data, mse_data, log_scale=True, name=None):
    """
    Creates separate publication-ready plots for correlation and MSE metrics.
    
    Parameters:
    -----------
    correlation_data : dict or str
        Either a dictionary with keys as method names and values as tuples of (x_values, y_means, y_stds)
        for correlation metrics, or a path to a CSV file containing correlation data
    mse_data : dict or str
        Either a dictionary with keys as method names and values as tuples of (x_values, y_means, y_stds)
        for MSE metrics, or a path to a CSV file containing MSE data
    log_scale : bool, optional
        Whether to use log scale for MSE plot (default: True)
    name : str, optional
        Base name for saving the plots (will append _correlation.pdf and _mse.pdf)
    """
    import seaborn as sns
    import pandas as pd
    
    # Set style consistent with publication quality
    sns.set_theme(style="ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 18
    
    # Process input data
    corr_data_dict = {}
    mse_data_dict = {}
    
    # Handle correlation data
    if isinstance(correlation_data, str):
        # It's a CSV file path
        df = pd.read_csv(correlation_data)
        
        # Check if it's in the format from the original plot_ablation function
        if 'N' in df.columns:
            # Extract method names from column names
            method_names = []
            for col in df.columns:
                if 'corr' in col and not col.endswith(('mean', 'std')):
                    method_name = col.split('_')[0]
                    if method_name == "mpinn":
                        method_name = r"$\mathcal{M}$-PINN"
                    elif method_name == "deltapinn":
                        method_name = r"$\Delta$-PINN"
                    method_names.append((method_name, col))
            
            # Group by N and calculate mean and std for each method
            grouped = df.groupby('N')
            
            for method_name, col in method_names:
                x_values = grouped['N'].first().values
                y_means = grouped[col].mean().values
                y_stds = grouped[col].std().values
                corr_data_dict[method_name] = (x_values, y_means, y_stds)
        else:
            # Assume it's already in the right format with columns: method, N, mean, std
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                corr_data_dict[method] = (method_df['N'].values, method_df['mean'].values, method_df['std'].values)
    else:
        # It's already a dictionary
        corr_data_dict = correlation_data
    
    # Handle MSE data
    if isinstance(mse_data, str):
        # It's a CSV file path
        df = pd.read_csv(mse_data)
        
        # Check if it's in the format from the original plot_ablation function
        if 'N' in df.columns:
            # Extract method names from column names
            method_names = []
            for col in df.columns:
                if 'mse' in col and not col.endswith(('mean', 'std')):
                    method_name = col.split('_')[0]
                    if method_name == "mpinn":
                        method_name = r"$\mathcal{M}$-PINN"
                    elif method_name == "deltapinn":
                        method_name = r"$\Delta$-PINN"
                    method_names.append((method_name, col))
            
            # Group by N and calculate mean and std for each method
            grouped = df.groupby('N')
            
            for method_name, col in method_names:
                x_values = grouped['N'].first().values
                y_means = grouped[col].mean().values
                y_stds = grouped[col].std().values
                mse_data_dict[method_name] = (x_values, y_means, y_stds)
        else:
            # Assume it's already in the right format with columns: method, N, mean, std
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                mse_data_dict[method] = (method_df['N'].values, method_df['mean'].values, method_df['std'].values)
    else:
        # It's already a dictionary
        mse_data_dict = mse_data
    
    # Define colors for different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot correlation
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    
    for i, (method_name, (x_values, y_means, y_stds)) in enumerate(corr_data_dict.items()):
        ax1.errorbar(
            x_values,
            y_means,
            yerr=y_stds,
            marker="o",
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=2,
            color=colors[i % len(colors)],
            label=method_name,
        )
    
    ax1.set_xlabel("number of train points")
    ax1.set_ylabel("correlation")
    ax1.grid(True, linestyle="--", alpha=0.6, linewidth=1.2)
    ax1.tick_params(axis="both", which="major", labelsize=18, width=1.5, length=6)
    ax1.legend(prop={"size": 18})
    
    plt.tight_layout()
    
    if name is not None:
        plt.savefig(f"{name}_correlation.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{name}_correlation.png", format="png", dpi=300, bbox_inches="tight")
    
    # Plot MSE
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    
    for i, (method_name, (x_values, y_means, y_stds)) in enumerate(mse_data_dict.items()):
        ax2.errorbar(
            x_values,
            y_means,
            yerr=y_stds,
            marker="o",
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=2,
            color=colors[i % len(colors)],
            label=method_name,
        )
    
    ax2.set_xlabel("number of train points")
    ax2.set_ylabel("MSE")
    ax2.grid(True, linestyle="--", alpha=0.6, linewidth=1.2)
    ax2.tick_params(axis="both", which="major", labelsize=18, width=1.5, length=6)
    ax2.legend(prop={"size": 18})
    
    if log_scale:
        ax2.set_yscale("log")
    
    plt.tight_layout()
    
    if name is not None:
        plt.savefig(f"{name}_mse.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{name}_mse.png", format="png", dpi=300, bbox_inches="tight")
    
    plt.show()
    
    return fig1, fig2






