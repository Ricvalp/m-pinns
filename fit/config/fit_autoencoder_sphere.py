from datetime import datetime

from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Charts to fit  # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts_to_fit = None # (0, 1)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Wandb  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "chart-autodecoder"
    cfg.wandb.name = "sphere"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_every_steps = 100
    cfg.wandb.log_charts_every = 3000

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Profiler # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.profiler = ConfigDict()
    cfg.profiler.start_step = 1000000
    cfg.profiler.end_step = 1000000
    cfg.profiler.log_dir = "./fit/profilier/sphere"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Checkpoint # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_path = "./fit/checkpoints/sphere"
    cfg.checkpoint.overwrite = True
    cfg.checkpoint.save_every = 100000  # Always save checkpoint at the end of training

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Model  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.model = ConfigDict()
    cfg.model.name = "AutoEncoder"
    cfg.model.n_hidden = 32
    cfg.model.n_latent = 2
    cfg.model.rff_dim = 128
    cfg.model.center = 0.5
    cfg.model.init_scale = 2.5

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.name = "Sphere"
    cfg.dataset.charts_path = "./datasets/sphere/charts"
    cfg.dataset.distance_matrix_path = "./datasets/sphere/charts/distance_matrix.npy"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Training  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.train = ConfigDict()
    cfg.train.batch_size = 128
    cfg.train.num_epochs = 300
    cfg.train.lr = 1e-4
    cfg.train.reg_lambda = 0.1
    cfg.train.weight_decay = 1e-3
    cfg.train.reg_lambda_decay = 0.9995
    cfg.train.reg = "reg+geo"
    cfg.train.noise_scale_riemannian = 0.02
    cfg.train.lambda_geo_loss = 5.0
    cfg.train.lambda_g_inv = 0.1
    return cfg
