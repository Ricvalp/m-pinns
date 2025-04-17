import gudhi as gd
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np


def process_compute_persistence_homology(chart):

    chart_id, chart = chart
    logging.info(f"Computing persistence homology for chart {chart_id}")

    subsampling = np.random.choice(chart.shape[0], 1000, replace=True)
    chart = chart[subsampling]

    complex = gd.AlphaComplex(points=chart)
    simplex_tree = complex.create_simplex_tree()

    simplex_tree.compute_persistence()

    logging.info(f"Chart {chart_id} done")

    return chart_id, simplex_tree.persistence()


def compute_persistence_homology(charts, name=None):

    persistence_diagrams = {}

    num_processes = mp.cpu_count() - 1
    with mp.Pool(processes=num_processes) as pool:
        logging.info(
            f"Calculating persistence homology using {num_processes} processes"
        )
        results = pool.map(
            process_compute_persistence_homology,
            charts.items(),
        )

        for chart_id, persistence_diagram in results:
            persistence_diagrams[chart_id] = persistence_diagram

    if name is not None:
        num_charts = len(persistence_diagrams)
        cols = 4
        rows = (num_charts + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for ax, (chart_id, persistence_diagram) in zip(
            axes, persistence_diagrams.items()
        ):
            gd.plot_persistence_diagram(persistence_diagram, axes=ax)
            ax.set_title(f"Chart {chart_id}")

        for ax in axes[num_charts:]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig(name)
        plt.close()

    return persistence_diagrams


if __name__ == "__main__":

    # Generate a 3D point cloud sampled from a cylinder
    theta = np.linspace(0, 2 * np.pi, 50)
    z = np.linspace(-1, 1, 50)
    theta, z = np.meshgrid(theta, z)
    x = np.cos(theta).flatten()
    y = np.sin(theta).flatten()
    z = z.flatten()
    points = np.vstack((x, y, z)).T

    charts = {
        0: points,
        1: points + np.random.normal(0, 0.05, points.shape),
        2: points + np.random.normal(0, 0.1, points.shape),
        3: points + np.random.normal(0, 0.2, points.shape),
        4: points + np.random.normal(0, 0.3, points.shape),
        5: points + np.random.normal(0, 0.4, points.shape),
        6: points + np.random.normal(0, 0.5, points.shape),
    }
    compute_persistence_homology(charts, name="test.png")
