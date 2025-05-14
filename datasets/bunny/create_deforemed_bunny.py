from absl import app, logging
from ml_collections import config_flags
from pathlib import Path
import numpy as np
from ml_collections import ConfigDict
from datetime import datetime
from datasets import get_dataset
import os

from chart_autoencoder import (
    plot_3d_charts,
    plot_local_charts_2d,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    get_charts,
    find_verts_in_charts,
    plot_3d_points,
    plot_3d_chart,
)
from chart_autoencoder import (
    get_umap_embeddings,
    compute_distance_matrix,
    save_charts,
)



def main(_):

    cfg = ConfigDict()
    cfg.seed = 37
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.seed = 37
    cfg.dataset.name = "DeformedBunny"
    cfg.dataset.t = 0.1
    cfg.dataset.path = "./datasets/bunny/stanford_bunny.obj"
    cfg.dataset.scale = 1.
    cfg.dataset.points_per_unit_area = 6
    cfg.dataset.subset_cardinality = None

    cfg.dataset.output_file = "./datasets/bunny/charts/t010/deformed_bunny_mesh.obj"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Charts  # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts = ConfigDict()
    cfg.charts.alg = "fast_region_growing"
    cfg.charts.min_dist = 6.0
    cfg.charts.nearest_neighbors = 10

    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)

    train_data = get_dataset(cfg.dataset)
    verts, connectivity = train_data.verts, train_data.connectivity

    # Create output filename for the OBJ file
    output_file = cfg.dataset.output_file
    
    # Save the mesh as an OBJ file
    save_obj(verts, connectivity, output_file)
    print(f"Saved mesh to {output_file}")

    plot_3d_points(
        points=train_data.data,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_points.png",
    )
    plot_3d_points(
        points=verts,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_verts.png",
    )


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def save_obj(verts, faces, filename):
    """
    Save vertices and faces to an OBJ file.

    Args:
        verts (numpy.ndarray): The vertices array
        faces (numpy.ndarray): The faces array
        filename (str): The path to save the file to
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory) and directory:
        os.makedirs(directory)

    with open(filename, "w") as f:
        f.write("# Generated sphere mesh\n")

        # Write vertices
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write faces (add 1 because OBJ format uses 1-based indices)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


if __name__ == "__main__":
    app.run(main)
