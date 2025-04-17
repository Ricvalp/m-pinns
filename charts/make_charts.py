from absl import app, logging
from ml_collections import config_flags
from pathlib import Path
import numpy as np
from datasets import get_dataset
from chart_autoencoder import (
    plot_3d_charts,
    plot_local_charts_2d,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    get_charts,
    find_verts_in_charts,
    plot_3d_points,
)
from chart_autoencoder import (
    get_umap_embeddings,
    compute_distance_matrix,
    save_charts,
)


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="charts/config/make_charts_coil.py"
)


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)

    train_data = get_dataset(cfg.dataset)
    verts, connectivity = train_data.verts, train_data.connectivity

    plot_3d_points(
        points=train_data.data,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_points.png",
    )

    logging.info(
        f"Loaded {cfg.dataset.name} dataset. Got {len(train_data.data)} points"
    )

    charts, charts_idxs, boundaries, boundary_indices, sampled_points = get_charts(
        points=train_data.data,
        charts_config=cfg.charts,
    )

    chart_in_mesh_indices, mesh_in_chart_indices = find_verts_in_charts(charts, verts)

    save_charts(
        charts_path=cfg.dataset.charts_path, 
        charts=charts,
        charts_idxs=charts_idxs,
        boundaries=boundaries,
        boundary_indices=boundary_indices,
        chart_in_mesh_indices=chart_in_mesh_indices,
        mesh_in_chart_indices=mesh_in_chart_indices,
        verts=verts,
        connectivity=connectivity
    )

    logging.info(f"Got {len(charts)} charts. Saved charts to {cfg.dataset.charts_path}")
    
    plot_html_3d_charts(
        charts=charts,
        sampled_points=sampled_points,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.html",
    )
    plot_html_3d_boundaries(
        boundaries=boundaries,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_boundaries.html",
    )
    plot_3d_charts(
        charts=charts,
        gt_charts=None,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.png",
    )

    distance_matrix = compute_distance_matrix(charts, cfg.charts.nearest_neighbors)
    np.save(cfg.dataset.distance_matrix_path, distance_matrix, allow_pickle=True)
    logging.info(f"Saved distance matrix to {cfg.dataset.distance_matrix_path}")

    umap_charts = get_umap_embeddings(charts, cfg)
    np.save(cfg.umap.umap_embeddings_path, umap_charts, allow_pickle=True)
    logging.info(f"Saved umap embeddings to {cfg.umap.umap_embeddings_path}")

    plot_local_charts_2d(
        charts=umap_charts,
        boundaries_indices=boundary_indices,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_umap_charts.png",
    )


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


if __name__ == "__main__":
    app.run(main)
