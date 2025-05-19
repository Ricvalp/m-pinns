import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go


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
        if i in bcs_x.keys():
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


def plot_domains_with_metric(x, y, sqrt_det_g, conditionings, name=None):
    num_plots = len(x)
    cols = 4
    rows = (num_plots + cols - 1) // cols

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_plots == 1:
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    for i in range(num_plots):
        row, col = divmod(i, cols)

        ax[row][col].set_title(f"Chart {i}")
        color_values = sqrt_det_g(conditionings[i], jnp.stack([x[i], y[i]], axis=1))
        scatter = ax[row][col].scatter(x[i], y[i], s=3, c=color_values, cmap="viridis")
        fig.colorbar(scatter, ax=ax[row][col], orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_3d(x, y, bcs_x, bcs_y, bcs, decoder, conditionings, d_params, charts_mu, charts_std, name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Combined 3D Plot")

    conditionings_list = [conditionings[i] for i in range(len(x))]

    # Create a single colorbar for all BC points
    all_bc_values = np.concatenate([bcs[i] for i in range(len(bcs)) if i in bcs_x.keys()])
    vmin, vmax = np.min(all_bc_values), np.max(all_bc_values)

    for i in range(len(x)):
        p = decoder.apply({"params": d_params}, np.stack([x[i], y[i]], axis=1), conditionings_list[i])
        if i in bcs_x.keys():
            p_bcs = decoder.apply({"params": d_params}, np.stack([bcs_x[i], bcs_y[i]], axis=1), conditionings_list[i])

        # Transform back the points
        p = p * charts_std[i] + charts_mu[i]
        if i in bcs_x.keys():
            p_bcs = p_bcs * charts_std[i] + charts_mu[i]

        # Plot the domain points
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=3, alpha=0.5, label=f"Chart {i}")

        # Plot the boundary conditions with colors
        if i in bcs_x.keys():
            scatter_bcs = ax.scatter(
                p_bcs[:, 0],
                p_bcs[:, 1],
                p_bcs[:, 2],
                c=bcs[i],
                s=50,
                vmin=vmin,
                vmax=vmax,
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


def plot_domains_3d_html(x, y, bcs_x, bcs_y, bcs, decoder, conditionings, d_params, charts_mu, charts_std, name=None):
    """
    Creates an interactive 3D plot of multiple domain charts and saves it as an HTML file.
    
    Args:
        x, y: Lists of x and y coordinates for each chart
        bcs_x, bcs_y: Lists of boundary condition x and y coordinates
        bcs: List of boundary condition values
        decoder: Decoder function
        conditionings: List of conditioning values
        d_params: Decoder parameters
        charts_mu, charts_std: Mean and standard deviation for each chart
        name (str, optional): File name to save the plot. Defaults to None.
    """
    # Create figure
    fig = go.Figure()
    
    conditionings_list = [conditionings[i] for i in range(len(x))]
    
    # Find global min and max for consistent colorscale
    all_bc_values = np.concatenate([bcs[i] for i in range(len(bcs)) if i in bcs_x.keys()])
    vmin, vmax = np.min(all_bc_values), np.max(all_bc_values)
    
    # Add domain points and boundary conditions for each chart
    for i in range(len(x)):
        # Decode the points
        p = decoder.apply({"params": d_params}, np.stack([x[i], y[i]], axis=1), conditionings_list[i])
        if i in bcs_x.keys():
            p_bcs = decoder.apply({"params": d_params}, np.stack([bcs_x[i], bcs_y[i]], axis=1), conditionings_list[i])
        
        # Transform back the points
        p = p * charts_std[i] + charts_mu[i]
        if i in bcs_x.keys():
            p_bcs = p_bcs * charts_std[i] + charts_mu[i]
        
        # Add domain points
        fig.add_trace(go.Scatter3d(
            x=p[:, 0],
            y=p[:, 1],
            z=p[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.5,
                color=f'rgb({50+40*i}, {100+30*i}, {150+20*i})'  # Different color for each chart
            ),
            name=f'Chart {i}'
        ))
        
        # Add boundary condition points
        if i in bcs_x.keys():
            fig.add_trace(go.Scatter3d(
                x=p_bcs[:, 0],
                y=p_bcs[:, 1],
                z=p_bcs[:, 2],
                mode='markers',
                marker=dict(
                size=5,
                color=bcs[i],
                colorscale='Viridis',
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title="BC Value")
                ),
                name=f'BCs {i}'
            ))
    
    # Set layout properties
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Combined 3D Domains"
    )
    
    # Save as HTML if name is provided
    if name is not None:
        fig.write_html(name)
    
    return fig


def plot_domains_3d_with_metric(x, y, decoder, sqrt_det_g, conditionings, d_params, name=None):
    num_plots = len(x)
    cols = 2
    rows = (num_plots + cols - 1) // cols

    fig = plt.figure(figsize=(15, 5 * rows))
    
    conditionings_list = [conditionings[i] for i in range(len(x))]

    for i in range(num_plots):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"Chart {i}")
        points_3d = decoder.apply(
            {"params": d_params}, jnp.stack([x[i], y[i]], axis=1), conditionings_list[i]
        )
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        color_values = sqrt_det_g(conditionings_list[i], jnp.stack([x[i], y[i]], axis=1))

        scatter = ax.scatter(x_3d, y_3d, z_3d, c=color_values, cmap="viridis", s=3)

        ax.legend(loc="best")
        fig.colorbar(scatter, ax=ax, orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_combined_3d_with_metric(x, y, decoder, sqrt_det_g, conditionings, d_params, charts_mu, charts_std, name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Combined 3D Plot with Metric Coloring")

    conditionings_list = [conditionings[i] for i in range(len(x))]

    for i in range(len(x)):
        # Decode the 2D points to 3D using the decoder function
        points_3d = decoder.apply(
            {"params": d_params}, jnp.stack([x[i], y[i]],  axis=1), conditionings_list[i]
        )
        points_3d = points_3d * charts_std[i] + charts_mu[i]
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        # Calculate the color values using sqrt_det_gs
        color_values = sqrt_det_g(conditionings_list[i], jnp.stack([x[i], y[i]], axis=1))

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


def plot_correlation(mesh_sol, gt_sol, data=None, name=None, min_val=0.0, max_val=4.0):
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
        c=(0, 0, 1),
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


def plot_ablation(mpinn_csv, deltapinn_csv, name=None):
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

    plt.tight_layout()

    if name is not None:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{name}.png", format="png", dpi=300, bbox_inches="tight")

    plt.show()
    return fig


def plot_charts_with_supernodes(chart, supernode_idxs, name=None):
    """
    Plot a batch of charts with the supernodes highlighted in a different color.
    
    Args:
        chart: Batch of 3D points representing charts
        supernode_idxs: Indices of supernodes for each chart
        distance_matrix: Distance matrix of the charts
        name: Optional path to save the figure
    """
    # Determine grid layout
    num_samples = min(16, len(chart))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    # Compute global min/max across all charts for consistent scaling
    x_min_global = np.min([chart[i][:, 0].min() for i in range(num_samples)])
    x_max_global = np.max([chart[i][:, 0].max() for i in range(num_samples)])
    y_min_global = np.min([chart[i][:, 1].min() for i in range(num_samples)])
    y_max_global = np.max([chart[i][:, 1].max() for i in range(num_samples)])
    z_min_global = np.min([chart[i][:, 2].min() for i in range(num_samples)])
    z_max_global = np.max([chart[i][:, 2].max() for i in range(num_samples)])
    
    # Add 10% padding to the global ranges
    x_pad = 0.1 * (x_max_global - x_min_global)
    y_pad = 0.1 * (y_max_global - y_min_global)
    z_pad = 0.1 * (z_max_global - z_min_global)
    
    # Compute global limits
    x_limits = [x_min_global - x_pad, x_max_global + x_pad]
    y_limits = [y_min_global - y_pad, y_max_global + y_pad]
    z_limits = [z_min_global - z_pad, z_max_global + z_pad]
    
    
    fig = plt.figure(figsize=(12, num_samples))
    
    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Create a mask for supernodes
        is_supernode = np.zeros(len(chart[i]), dtype=bool)
        is_supernode[supernode_idxs[i]] = True
        
        # Plot regular points with consistent colormap scaling
        ax.scatter(
            chart[i][~is_supernode, 0],
            chart[i][~is_supernode, 1],
            chart[i][~is_supernode, 2],
            alpha=0.5,
            s=10,
            label='Regular Points',
        )
        
        # Plot supernodes with different color
        ax.scatter(
            chart[i][is_supernode, 0],
            chart[i][is_supernode, 1],
            chart[i][is_supernode, 2],
            c='red',
            alpha=1.0,
            s=30,
            label='Supernodes'
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Use global limits for all plots
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.legend()
    
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, dpi=300)
    plt.close()


def plot_3d_point_cloud(points, colors=None, labels=None, marker_size=50, view_angles=(30, 45), 
                        colormap='viridis', alpha=1.0, name=None, figsize=(10, 8), 
                        colorbar_label=None, point_labels=None, axis_labels=None):
    """
    Creates a publication-quality 3D point cloud visualization.
    
    Args:
        points (np.ndarray): Array of shape (n, 3) containing 3D point coordinates.
        colors (np.ndarray, optional): Array of values to color the points by. Defaults to None.
        labels (dict, optional): Dictionary mapping category names to boolean masks for points. Defaults to None.
        marker_size (int or np.ndarray, optional): Size of markers. Can be an array for variable sizes. Defaults to 50.
        view_angles (tuple, optional): Tuple of (elevation, azimuth) viewing angles. Defaults to (30, 45).
        colormap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
        alpha (float, optional): Transparency of points (0 to 1). Defaults to 1.0.
        name (str, optional): Base name for saving the plot files. Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to (10, 8).
        colorbar_label (str, optional): Label for the colorbar. Defaults to None.
        point_labels (list, optional): List of text labels for specific points. Defaults to None.
        axis_labels (list, optional): Custom labels for x, y, z axes. Defaults to None.
    
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
    plt.rcParams["legend.fontsize"] = 18
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Main scatter plot
    if labels is not None:
        # Plot each category with a different color
        for label_name, mask in labels.items():
            ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                s=marker_size, 
                label=label_name,
                alpha=alpha,
                edgecolor='none'
            )
        ax.legend(prop={'size': 18}, markerscale=1.5)
    elif colors is not None:
        # Plot with continuous color mapping
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, 
            cmap=colormap,
            s=marker_size,
            alpha=alpha,
            edgecolor='none'
        )
        
        # Add colorbar with professional styling
        if np.unique(colors).size > 1:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
            if colorbar_label:
                cbar.set_label(colorbar_label, size=18)
            cbar.ax.tick_params(labelsize=12)
    else:
        # Simple scatter plot without color mapping
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=marker_size,
            alpha=alpha,
            edgecolor='none'
        )
    
    # Add point labels if provided
    if point_labels is not None:
        for i, txt in enumerate(point_labels):
            if txt:  # Only label non-empty strings
                ax.text(points[i, 0], points[i, 1], points[i, 2], txt, 
                       size=14, zorder=100, color='black', fontweight='bold')
    
    # Set axis labels
    if axis_labels:
        ax.set_xlabel(axis_labels[0], labelpad=10)
        ax.set_ylabel(axis_labels[1], labelpad=10)
        ax.set_zlabel(axis_labels[2], labelpad=10)
    else:
        ax.set_xlabel('X', labelpad=10)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Z', labelpad=10)
    
    # Professional styling for the plot
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    
    # Set viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set equal aspect ratio for all axes
    # This ensures the 3D visualization isn't stretched
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust tick parameters for better readability
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6, pad=8)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save high-quality versions if name is provided
    if name is not None:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{name}.png", format="png", dpi=600, bbox_inches="tight")
    
    plt.show()
    return fig


def plot_3d_point_cloud_single_chart(points, colors=None, marker_size=100, view_angles=(30, 45), 
                        colormap='viridis', alpha=1.0, name=None, figsize=(10, 8), 
                        colorbar_label=None, edge_color=None, edge_width=0,
                        show_axes=True, axes_linewidth=1.5, axes_color='black'):
    """
    Creates a minimal, publication-quality 3D point cloud visualization with optional axes.
    
    Args:
        points (np.ndarray): Array of shape (n, 3) containing 3D point coordinates.
        colors (np.ndarray, optional): Array of values to color the points by. Defaults to None.
        marker_size (int or np.ndarray, optional): Size of markers. Can be an array for variable sizes. Defaults to 50.
        view_angles (tuple, optional): Tuple of (elevation, azimuth) viewing angles. Defaults to (30, 45).
        colormap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
        alpha (float, optional): Transparency of points (0 to 1). Defaults to 1.0.
        name (str, optional): Base name for saving the plot files. Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to (10, 8).
        colorbar_label (str, optional): Label for the colorbar. Defaults to None.
        edge_color (str, optional): Color of point edges. None for no edges. Defaults to None.
        edge_width (float, optional): Width of point edges. Defaults to 0.
        show_axes (bool, optional): Whether to show coordinate axes. Defaults to True.
        axes_linewidth (float, optional): Width of coordinate axes lines. Defaults to 1.5.
        axes_color (str, optional): Color of coordinate axes. Defaults to 'black'.
    
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    # Set a professional style using seaborn
    sns.set_theme(style="ticks")
    plt.rcParams["font.family"] = "serif"  # Use a serif font for better readability in papers
    plt.rcParams["font.size"] = 10
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Calculate axis limits with a small buffer
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # Add a small buffer (5% of range) to avoid points at the edges
    buffer_x = 0.05 * (x_max - x_min)
    buffer_y = 0.05 * (y_max - y_min)
    buffer_z = 0.05 * (z_max - z_min)
    
    ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
    ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
    ax.set_zlim(z_min - buffer_z, z_max + buffer_z)
    
    # Remove all default axis elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Remove background panes and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    # Set viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    

    # Add minimal coordinate axes if requested
    if show_axes:
        # Get the minimum values for each axis to place the coordinate system
        origin = (x_min - buffer_x/2, y_min - buffer_y/2, z_min - buffer_z/2)

    # Set edge color parameters
    if edge_color is None:
        edge_color = 'none'
    
    # Main scatter plot
    if colors is not None:
        # Plot with continuous color mapping
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, 
            cmap=colormap,
            s=marker_size,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=edge_width
        )
        
        # Add colorbar with minimal styling if needed
        if np.unique(colors).size > 1:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
            if colorbar_label:
                cbar.set_label(colorbar_label, size=18)
            cbar.ax.tick_params(labelsize=12)
    else:
        # Simple scatter plot without color mapping
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=marker_size,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=edge_width
        )

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    # Tight layout with minimal padding
    plt.tight_layout(pad=0.1)
    
    # Save high-quality versions if name is provided
    if name is not None:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{name}.png", format="png", dpi=600, bbox_inches="tight")
    
    plt.show()
    return fig


def plot_3d_point_cloud_single_chart(points, colors=None, marker_size=50, view_angles=(30, 45), 
                        colormap='viridis', alpha=0.8, name=None, figsize=(10, 8), 
                        colorbar_label=None, edge_color='black', edge_width=0.3,
                        show_axes=True, axes_linewidth=1.5, axes_color='black',
                        depth_shading=True, elev_range=15, azim_range=45, n_views=None):
    """
    Creates a publication-quality 3D point cloud visualization with enhanced depth perception.
    
    Args:
        points (np.ndarray): Array of shape (n, 3) containing 3D point coordinates.
        colors (np.ndarray, optional): Array of values to color the points by. Defaults to None.
        marker_size (int or np.ndarray, optional): Size of markers. Can be an array for variable sizes. Defaults to 50.
        view_angles (tuple, optional): Tuple of (elevation, azimuth) viewing angles. Defaults to (30, 45).
        colormap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
        alpha (float, optional): Transparency of points (0 to 1). Defaults to 0.8.
        name (str, optional): Base name for saving the plot files. Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to (10, 8).
        colorbar_label (str, optional): Label for the colorbar. Defaults to None.
        edge_color (str, optional): Color of point edges. Defaults to 'black'.
        edge_width (float, optional): Width of point edges. Defaults to 0.3.
        show_axes (bool, optional): Whether to show coordinate axes. Defaults to True.
        axes_linewidth (float, optional): Width of coordinate axes lines. Defaults to 1.5.
        axes_color (str, optional): Color of coordinate axes. Defaults to 'black'.
        depth_shading (bool, optional): Whether to apply depth-dependent shading. Defaults to True.
        elev_range (float, optional): Range of elevation angles for multi-view plots. Defaults to 15.
        azim_range (float, optional): Range of azimuth angles for multi-view plots. Defaults to 45.
        n_views (int, optional): Number of views to generate. If None, creates a single view. Defaults to None.
    
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    # Set a professional style using seaborn
    sns.set_theme(style="ticks")
    plt.rcParams["font.family"] = "serif"  # Use a serif font for better readability in papers
    plt.rcParams["font.size"] = 10
    
    # For multi-view plot
    if n_views is not None and n_views > 1:
        fig = plt.figure(figsize=(figsize[0] * min(n_views, 3), figsize[1] * ((n_views + 2) // 3)))
        
        # Calculate view angles for multiple views
        base_elev, base_azim = view_angles
        elevs = np.linspace(base_elev - elev_range/2, base_elev + elev_range/2, n_views)
        azims = np.linspace(base_azim - azim_range/2, base_azim + azim_range/2, n_views)
        
        for i in range(n_views):
            ax = fig.add_subplot(((n_views + 2) // 3), min(n_views, 3), i+1, projection='3d')
            _create_3d_plot(ax, points, colors, marker_size, (elevs[i], azims[i]), 
                           colormap, alpha, colorbar_label, edge_color, edge_width,
                           show_axes, axes_linewidth, axes_color, depth_shading)
            
        plt.tight_layout()
        
    else:  # Single view plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        _create_3d_plot(ax, points, colors, marker_size, view_angles, 
                       colormap, alpha, colorbar_label, edge_color, edge_width,
                       show_axes, axes_linewidth, axes_color, depth_shading)
    
    # Save high-quality versions if name is provided
    if name is not None:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{name}.png", format="png", dpi=600, bbox_inches="tight")
    
    plt.show()
    return fig


def _create_3d_plot(ax, points, colors, marker_size, view_angles, colormap, alpha, 
                   colorbar_label, edge_color, edge_width, show_axes, 
                   axes_linewidth, axes_color, depth_shading):
    """Helper function to create a 3D plot with enhanced depth perception."""
    
    # Calculate axis limits with a small buffer
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # Add a small buffer (5% of range) to avoid points at the edges
    buffer_x = 0.05 * (x_max - x_min)
    buffer_y = 0.05 * (y_max - y_min)
    buffer_z = 0.05 * (z_max - z_min)
    
    # Apply depth-dependent shading if requested
    if depth_shading and colors is None:
        # Calculate depth values based on the view angle
        elev, azim = view_angles
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        
        # Direction vector from viewer to origin
        view_dir = np.array([
            -np.cos(elev_rad) * np.sin(azim_rad),
            -np.cos(elev_rad) * np.cos(azim_rad),
            -np.sin(elev_rad)
        ])
        
        # Project points onto viewing direction to get depth
        depths = np.dot(points, view_dir)
        
        # Normalize depths to [0, 1] for coloring
        depth_min, depth_max = depths.min(), depths.max()
        norm_depths = (depths - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depths)
        
        # Use depth for coloring
        colors = norm_depths
        colormap = 'Blues_r'  # Reversed blues colormap works well for depth
    
    # Main scatter plot with edge highlighting for better 3D appearance
    if colors is not None:
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, 
            cmap=colormap,
            s=marker_size,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=edge_width,
            depthshade=True  # Enable matplotlib's built-in depth shading
        )
        
        # Add colorbar with minimal styling if needed
        if np.unique(colors).size > 1:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
            if colorbar_label:
                cbar.set_label(colorbar_label, size=14)
            cbar.ax.tick_params(labelsize=10)
    else:
        # Simple scatter plot without color mapping
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=marker_size,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=edge_width,
            depthshade=True  # Enable matplotlib's built-in depth shading
        )
    
    # Remove all default axis elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Add subtle edge to the 3D box for better depth perception
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Add minimal coordinate axes if requested
    if show_axes:
        # Get the minimum values for each axis to place the coordinate system
        origin = (x_min - buffer_x/2, y_min - buffer_y/2, z_min - buffer_z/2)
        
        # Calculate axis lengths (20% of the respective dimension)
        x_length = 0.2 * (x_max - x_min)
        y_length = 0.2 * (y_max - y_min)
        z_length = 0.2 * (z_max - z_min)
        
        # Draw x-axis
        ax.plot([origin[0], origin[0] + x_length], 
                [origin[1], origin[1]], 
                [origin[2], origin[2]], 
                color=axes_color, linewidth=axes_linewidth)
        
        # Draw y-axis
        ax.plot([origin[0], origin[0]], 
                [origin[1], origin[1] + y_length], 
                [origin[2], origin[2]], 
                color=axes_color, linewidth=axes_linewidth)
        
        # Draw z-axis
        ax.plot([origin[0], origin[0]], 
                [origin[1], origin[1]], 
                [origin[2], origin[2] + z_length], 
                color=axes_color, linewidth=axes_linewidth)
    
    # Set viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    # Set axis limits
    ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
    ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
    ax.set_zlim(z_min - buffer_z, z_max + buffer_z)


def plot_2d_scatter(x, y, marker_size=50, colormap='viridis', alpha=0.8, 
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
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add a small buffer (5% of range) to avoid points at the edges
    buffer_x = 0.05 * (x_max - x_min)
    buffer_y = 0.05 * (y_max - y_min)
    
    # Set axis limits
    ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
    ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
    

    scatter = ax.scatter(
        x, y,
        s=marker_size,
        alpha=alpha,
        edgecolor=edge_color,
        linewidth=edge_width,
        c='b'
    )
    
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
