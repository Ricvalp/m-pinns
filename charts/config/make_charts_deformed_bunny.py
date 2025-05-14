from datetime import datetime

from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.seed = 37
    cfg.dataset.name = "StanfordBunny"
    cfg.dataset.path = "./datasets/bunny/charts/t015/deformed_bunny_mesh.obj"
    cfg.dataset.scale = 1.0
    cfg.dataset.points_per_unit_area = 6
    cfg.dataset.subset_cardinality = None
    cfg.dataset.charts_path = "./datasets/bunny/charts/t015/"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Charts  # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts = ConfigDict()
    cfg.charts.alg = "fast_region_growing"
    cfg.charts.min_dist = 6.0
    cfg.charts.nearest_neighbors = 10

    return cfg
