from datetime import datetime

from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  UMAP # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.umap = ConfigDict()
    cfg.umap.umap_scale = 0.1
    cfg.umap.umap_embeddings_path = "./datasets/bunny/charts/umap_embeddings.npy"
    cfg.umap.n_neighbors = 15
    cfg.umap.learning_rate = 1.0
    cfg.umap.min_dist = 0.8
    cfg.umap.random_state = 42
    cfg.umap.n_components = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.seed = 37
    cfg.dataset.name = "StanfordBunny"
    cfg.dataset.path = "./datasets/bunny/stanford_bunny.obj"
    cfg.dataset.scale = 0.1
    cfg.dataset.points_per_unit_area = 5
    cfg.dataset.subset_cardinality = 300000
    cfg.dataset.charts_path = "./datasets/bunny/charts"
    cfg.dataset.distance_matrix_path = "./datasets/bunny/charts/distance_matrix.npy"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Charts  # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts = ConfigDict()
    cfg.charts.alg = "fast_region_growing"
    cfg.charts.min_dist = 1.
    cfg.charts.nearest_neighbors = 10

    return cfg
