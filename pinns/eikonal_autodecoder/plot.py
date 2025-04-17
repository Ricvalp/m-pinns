import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


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
            x[key], y[key], c=u_preds[key], cmap="jet", s=100., vmin=vmin, vmax=vmax
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


def plot_correlation(mesh_sol, gt_sol, data=None, name=None):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mesh_sol, gt_sol, s=5)
    if data is not None:
        ax.scatter(data, data, s=25, c="red")
    ax.set_xlabel("Mesh Solution")
    ax.set_ylabel("Ground Truth Solution")
    ax.set_title("Correlation between Mesh and GT Solutions")

    # Add a diagonal line for reference
    min_val = min(np.min(mesh_sol), np.min(gt_sol))
    max_val = max(np.max(mesh_sol), np.max(gt_sol))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

    if name is not None:
        plt.savefig(name)
    plt.show()

    return fig
