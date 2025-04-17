from absl import app, logging
from ml_collections import config_flags
from pathlib import Path
import numpy as np
from chart_autoencoder import (
    plot_3d_charts,
    plot_local_charts_2d,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    load_charts,
    compute_persistence_homology,
)


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="charts/config/make_charts_coil.py"
)


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)

    charts, boundaries, boundary_indices, _ = load_charts(
        charts_path=cfg.dataset.charts_path,
    )

    compute_persistence_homology(
        charts=charts,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_persistence_homology.png",
    )

    logging.info(f"Got {len(charts)} charts")

    plot_html_3d_charts(
        charts=charts,
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

    umap_charts = np.load(cfg.umap.umap_embeddings_path, allow_pickle=True).item()
    logging.info("Loaded UMAP embeddings")

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
