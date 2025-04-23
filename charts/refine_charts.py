from absl import app, logging
from ml_collections import config_flags
from pathlib import Path
import numpy as np
import sys
from chart_autoencoder import (
    plot_3d_charts,
    plot_local_charts_2d,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    refine_chart,
    reindex_charts,
    load_charts,
    save_charts,
)
from chart_autoencoder.umap_embedding import get_umap_embeddings
from chart_autoencoder.utils import compute_distance_matrix


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="charts/config/refine_charts_coil.py"
)


def main(_):

    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)

    charts, charts_idxs, _, _, _ = load_charts(
        charts_path=cfg.dataset.charts_path,
    )

    plot_html_3d_charts(
        charts=charts,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_old_charts.html",
    )
    plot_3d_charts(
        charts=charts,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_old_charts.png",
    )

    chart_to_refine = cfg.charts_to_refine.chart_to_refine
    refined_chart, refined_idxs, refined_sampled_points = refine_chart(
        points=charts[chart_to_refine],
        points_idxs=charts_idxs[chart_to_refine],
        charts_to_refine_cfg=cfg.charts_to_refine,
    )

    plot_html_3d_charts(
        charts=refined_chart,
        sampled_points=refined_sampled_points,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_refined_charts.html",
    )
    plot_3d_charts(
        charts=refined_chart,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_refined_charts.png",
    )

    (
        reindexed_charts,
        reindexed_idxs,
        reindexed_boundaries,
        reindexed_boundary_indices,
    ) = reindex_charts(
        old_charts=charts,
        old_idxs=charts_idxs,
        key_chart_to_refine=chart_to_refine,
        refined_charts=refined_chart,
        refined_idxs=refined_idxs,
    )

    plot_html_3d_charts(
        charts=reindexed_charts,
        sampled_points=refined_sampled_points,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_reindexed_charts.html",
    )

    plot_html_3d_boundaries(
        boundaries=reindexed_boundaries,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_reindexed_boundaries.html",
    )
    plot_3d_charts(
        charts=reindexed_charts,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_reindexed_charts.png",
    )

    distance_matrix = compute_distance_matrix(
        reindexed_charts, cfg.charts.nearest_neighbors
    )

    umap_charts = get_umap_embeddings(reindexed_charts, cfg)

    plot_local_charts_2d(
        charts=umap_charts,
        boundaries_indices=reindexed_boundary_indices,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_umap_charts.png",
    )

    user_input = input("Do you want to save the file? (yes/no): ").strip().lower()

    if user_input in ["yes", "y"]:
        save_charts(
            cfg.dataset.charts_path,
            reindexed_charts,
            reindexed_idxs,
            reindexed_boundaries,
            reindexed_boundary_indices,
        )
        logging.info(f"Saved charts to {cfg.dataset.charts_path}")

        np.save(cfg.dataset.distance_matrix_path, distance_matrix, allow_pickle=True)
        logging.info(f"Saved distance matrix to {cfg.dataset.distance_matrix_path}")

        np.save(cfg.umap.umap_embeddings_path, umap_charts, allow_pickle=True)
        logging.info(f"Saved UMAP embeddings to {cfg.umap.umap_embeddings_path}")


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


if __name__ == "__main__":
    app.run(main)
