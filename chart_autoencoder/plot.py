import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import numpy as np


def plot_3d_points(points, name=None, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], **kwargs)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect([1, 1, 1])

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_3d_charts(charts, gt_charts=None, name=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("viridis", len(charts))
    colors = [cmap(j / len(charts)) for j in range(len(charts))]

    for key, color in zip(charts.keys(), colors):
        ax.scatter(
            charts[key][:, 0],
            charts[key][:, 1],
            charts[key][:, 2],
            color=color,
        )

    if gt_charts is not None:
        for key in gt_charts.keys():
            ax.scatter(
                gt_charts[key][:, 0],
                gt_charts[key][:, 1],
                gt_charts[key][:, 2],
                color="red",
            )

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_3d_boundaries(
    boundaries, elev=30, azim=30, set_lims=None, name=None, i=0, **kwargs
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("viridis", len(boundaries))
    colors = [cmap(j / len(boundaries)) for j in range(len(boundaries))]

    for points, color in zip(boundaries.values(), colors):
        ax.scatter(
            points[:, i], points[:, i + 1], points[:, i + 2], color=color, **kwargs
        )

    if set_lims is not None:
        ax.set_xlim([-set_lims, set_lims])
        ax.set_ylim([-set_lims, set_lims])
        ax.set_zlim([-set_lims, set_lims])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=elev, azim=azim)
    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_html_3d_point_cloud(points, name=None):
    fig_plotly = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(color="blue", size=2),
            )
        ]
    )
    fig_plotly.write_html(name)


def plot_html_3d_charts(
    charts, original_charts=None, sampled_points=None, g=None, name=None
):

    cmap = plt.get_cmap("viridis", len(charts))
    rgbs = [cmap(int(key) / len(charts))[:3] for key in charts.keys()]
    colors = {
        key: f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"
        for key, (r, g, b) in zip(charts.keys(), rgbs)
    }

    data = []

    if g is not None:
        for idx, key in enumerate(charts.keys()):
            data.append(
                go.Scatter3d(
                    x=charts[key][:, 0],
                    y=charts[key][:, 1],
                    z=charts[key][:, 2],
                    mode="markers",
                    marker=dict(
                        color=g[key],
                        size=2,
                        colorscale="Viridis",
                        colorbar=dict(title="Metric") if idx == 0 else None,
                        showscale=idx == 0,
                    ),
                    text=f"Chart {key}",
                )
            )
    else:
        for idx, key in enumerate(charts.keys()):
            data.append(
                go.Scatter3d(
                    x=charts[key][:, 0],
                    y=charts[key][:, 1],
                    z=charts[key][:, 2],
                    mode="markers",
                    marker=dict(color=colors[key], size=2),
                    text=f"Chart {key}",
                )
            )

    if original_charts is not None:
        for key in original_charts.keys():
            data.append(
                go.Scatter3d(
                    x=original_charts[key][:, 0],
                    y=original_charts[key][:, 1],
                    z=original_charts[key][:, 2],
                    mode="markers",
                    marker=dict(color="red", size=2),
                    text=f"Original Chart {key}",
                )
            )

    if sampled_points is not None:
        data.append(
            go.Scatter3d(
                x=sampled_points[:, 0],
                y=sampled_points[:, 1],
                z=sampled_points[:, 2],
                mode="markers",
                marker=dict(color="red", size=8),
            )
        )

    fig_plotly = go.Figure(data=data)
    fig_plotly.write_html(name)


def plot_html_3d_boundaries(boundaries, name=None):
    cmap = plt.get_cmap("viridis", len(boundaries))
    colors = [cmap(j / len(boundaries)) for j in range(len(boundaries))]

    fig_plotly = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(color=color, size=2),
            )
            for points, color in zip(boundaries.values(), colors)
        ]
    )
    fig_plotly.write_html(name)


def plot_local_charts_2d_with_boundaries(charts, boundaries, name=None, **kwargs):
    num_charts = len(charts)
    num_cols = int(math.ceil(math.sqrt(num_charts)))
    num_rows = int(math.ceil(num_charts / num_cols))

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(10, 10), sharex=True, sharey=True
    )
    axs = axs.flatten()

    for i, (ax, points) in enumerate(zip(axs, zip(charts, boundaries))):
        points, boundary = points
        ax.scatter(points[:, 0], points[:, 1], c="b", **kwargs)
        ax.scatter(boundary[:, 0], boundary[:, 1], c="r", **kwargs)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_local_charts_2d(
    charts, boundaries_indices, original_charts=None, g=None, name=None
):
    num_charts = len(charts)
    num_cols = int(math.ceil(math.sqrt(num_charts)))
    num_rows = int(math.ceil(num_charts / num_cols))

    if num_charts == 1:
        fig, ax = plt.subplots(figsize=(10, 10), sharex=True, sharey=True)
        axs = [ax]
    else:
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(10, 10), sharex=True, sharey=True
        )
        axs = axs.flatten()

    colors = plt.cm.tab20(
        np.linspace(0, 1, len(charts) + 1)
    ).tolist()  # Creates 20 distinct colors that will cycle

    if g is not None:
        for i, (ax, key) in enumerate(zip(axs, charts.keys())):
            ax.scatter(charts[key][:, 0], charts[key][:, 1], c=g[key], s=0.1)
            if original_charts is not None:
                ax.scatter(
                    original_charts[key][:, 0],
                    original_charts[key][:, 1],
                    c="r",
                    s=0.2,
                )
            ax.set_aspect("equal", "box")
            ax.set_title(f"Chart {key}")

            for j, k in boundaries_indices.keys():
                if j == key:
                    ax.scatter(
                        charts[key][boundaries_indices[j, k], 0],
                        charts[key][boundaries_indices[j, k], 1],
                        s=0.1,
                        label=f"Boundary {i} to {k}",
                    )
            scatter = ax.scatter(charts[key][:, 0], charts[key][:, 1], c=g[key], s=0.1)
            plt.colorbar(scatter, ax=ax)

    else:
        for i, (ax, key) in enumerate(zip(axs, charts.keys())):
            ax.scatter(charts[key][:, 0], charts[key][:, 1], c="b", s=0.1)
            if original_charts is not None:
                ax.scatter(
                    original_charts[key][:, 0], original_charts[key][:, 1], c="r", s=0.2
                )
            ax.set_aspect("equal", "box")
            ax.set_title(f"Chart {key}")
            for j, k in boundaries_indices.keys():
                if j == key:
                    ax.scatter(
                        charts[key][boundaries_indices[j, k], 0],
                        charts[key][boundaries_indices[j, k], 1],
                        s=0.1,
                        label=f"Boundary {i} to {k}",
                    )

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    if name is not None:
        plt.savefig(name)
    plt.close()


def plot_html_3d_mesh(vertices, faces, name=None, colors=None, colorscale="Viridis"):
    """
    Plot a 3D mesh with triangles and save it as an HTML file.

    Args:
        vertices: numpy array of shape (n_vertices, 3) containing the x, y, z coordinates
        faces: numpy array of shape (n_faces, 3) containing the indices of vertices that form triangles
        name: output HTML file path
        colors: optional array of values to color the mesh by (per vertex or per face)
        colorscale: colorscale name for Plotly (default: "Viridis")
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                intensity=colors,
                colorscale=colorscale,
                opacity=0.8,
                showscale=colors is not None,
            )
        ]
    )

    # Add wireframe to see the edges more clearly
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
        )
    )

    # Create lists for wireframe lines
    wireframe_x, wireframe_y, wireframe_z = [], [], []
    for triangle in faces:
        for i in range(3):
            # Get vertex indices for the edge
            idx1, idx2 = triangle[i], triangle[(i + 1) % 3]
            # Add the vertices of the edge
            wireframe_x.extend([vertices[idx1, 0], vertices[idx2, 0], None])
            wireframe_y.extend([vertices[idx1, 1], vertices[idx2, 1], None])
            wireframe_z.extend([vertices[idx1, 2], vertices[idx2, 2], None])

    # Add the wireframe
    fig.add_trace(
        go.Scatter3d(
            x=wireframe_x,
            y=wireframe_y,
            z=wireframe_z,
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
        )
    )

    # Set layout
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            zaxis=dict(showgrid=True),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    if name is not None:
        fig.write_html(name)
    return fig


def plot_3d_charts(charts, gt_charts=None, name=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("viridis", len(charts))
    colors = [cmap(j / len(charts)) for j in range(len(charts))]

    for key, color in zip(charts.keys(), colors):
        ax.scatter(
            charts[key][:, 0],
            charts[key][:, 1],
            charts[key][:, 2],
            color=color,
        )

    if gt_charts is not None:
        for key in gt_charts.keys():
            ax.scatter(
                gt_charts[key][:, 0],
                gt_charts[key][:, 1],
                gt_charts[key][:, 2],
                color="red",
            )

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if name is not None:
        plt.savefig(name)
    plt.show()



###############################################
###############################################
############## Publication plots ##############
###############################################
###############################################




def plot_3d_chart(chart, name=None, figsize=(10, 8), dpi=300, 
                   marker_size=100, alpha=0.7, tick_size=12):
    """
    Create a publication-ready 3D scatter plot of chart points without axis labels.
    
    Parameters:
    -----------
    chart : array
        Array of 3D points to plot
    name : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution for saved figure
    marker_size : int, optional
        Size of scatter points
    alpha : float, optional
        Transparency of points (0-1)
    legend : bool, optional
        Whether to show legend
    palette : str, optional
        Seaborn color palette name
    """
    import seaborn as sns
    
    chart = (chart - chart.mean(axis=0)) / chart.std()


    # Set the seaborn style for publication quality
    sns.set_theme(style="whitegrid", context="paper")
    
    # Create figure with tight layout
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        chart[:, 0],
        chart[:, 1],
        chart[:, 2],
        color='b',
        s=marker_size,
        alpha=alpha,
        edgecolor='w',
        linewidth=0.5,
    )

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    
    # Reduce number of ticks on each axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=tick_size, pad=6)
    
    # Add grid for better readability but make it less prominent
    ax.grid(True, linestyle='--', alpha=0.4)

    # Remove background panes to reduce visual clutter
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges less prominent
    ax.xaxis.pane.set_edgecolor('lightgrey')
    ax.yaxis.pane.set_edgecolor('lightgrey')
    ax.zaxis.pane.set_edgecolor('lightgrey')
        
    # Save the figure if a name is provided
    if name is not None:
        plt.savefig(name, bbox_inches='tight', dpi=dpi)
    
    plt.show()


