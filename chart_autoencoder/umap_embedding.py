import umap
import multiprocessing as mp
from functools import partial
import logging


def process_chart_umap(chart, umap_embedder, scale):
    chart_id, chart = chart
    logging.info(f"Starting UMAP on chart {chart_id} with shape {chart.shape}")
    embedding = umap_embedder.fit_transform(chart) * scale
    logging.info(f"Finished UMAP on chart {chart_id} with shape {embedding.shape}")
    return chart_id, embedding


def get_umap_embeddings(charts, cfg):
    """
    Get UMAP embeddings for all charts.

    Args:
        charts: Dict[str, np.ndarray]
        cfg: OmegaConf

    Returns:
        Dict
    """

    umap_embedder = umap.UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        learning_rate=cfg.umap.learning_rate,
        min_dist=cfg.umap.min_dist,
        random_state=cfg.umap.random_state,
        n_components=cfg.umap.n_components,
    )
    umap_charts = {}

    num_processes = mp.cpu_count() - 1
    with mp.Pool(processes=num_processes) as pool:
        logging.info(f"Calculating UMAP embeddings using {num_processes} processes")
        results = pool.map(
            partial(
                process_chart_umap,
                umap_embedder=umap_embedder,
                scale=cfg.umap.umap_scale,
            ),
            charts.items(),
        )

        for chart_id, embedding in results:
            umap_charts[chart_id] = embedding

    return umap_charts
