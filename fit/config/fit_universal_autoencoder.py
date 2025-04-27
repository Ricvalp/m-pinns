from datetime import datetime

from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Wandb  # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "universal-autoencoder"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_every_steps = 100
    cfg.wandb.log_charts_every = 1000  # Frequency for logging chart visualizations

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
    cfg.checkpoint.checkpoint_path = "./fit/checkpoints/universal_autoencoder"
    cfg.checkpoint.overwrite = True
    cfg.checkpoint.save_every = 100000  # Always save checkpoint at the end of training

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Model  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.model = ConfigDict()
    cfg.model.name = "UniversalAutoencoder"
    # Input dimensions and structure parameters
    cfg.model.input_dim = 3  # Input coordinate dimensions (e.g., 3 for 3D points)
    cfg.model.ndim = 3  # Number of spatial dimensions
    # Graph neural network parameters
    cfg.model.radius = 1.0  # Radius for neighbor search
    cfg.model.max_degree = 5  # Maximum number of neighbors per node
    cfg.model.gnn_dim = 128  # GNN hidden dimension
    # Encoder parameters
    cfg.model.enc_dim = 128  # Encoder hidden dimension
    cfg.model.enc_depth = 4  # Number of transformer layers in encoder
    cfg.model.enc_num_heads = 4  # Number of attention heads in encoder
    # Perceiver parameters
    cfg.model.perc_dim = 64  # Perceiver hidden dimension
    cfg.model.perc_num_heads = 4  # Number of attention heads in perceiver
    cfg.model.num_latent_tokens = 8  # Number of latent tokens in perceiver
    # Conditioning parameters
    cfg.model.cond_dim = None  # Dimension of condition vector (None if not using conditioning)
    cfg.model.cond_encoder_features = (128, 128)  # Condition encoder hidden dimensions
    # SIREN parameters
    cfg.model.siren_features = (256, 256, 256, 3)  # SIREN network hidden dimensions
    cfg.model.w0 = 30.0  # Frequency for SIREN hidden layers
    cfg.model.w0_initial = 30.0  # Frequency for SIREN first layer
    cfg.model.coord_dim = 2  # Dimension of coordinates for SirenModel
    # Coordinate transformation parameters
    cfg.model.output_coord_dim = 2  # Output coordinate dimensions (2D latent space)
    cfg.model.coord_enc_dim = 64  # Dimension for coordinate encoder
    cfg.model.coord_enc_depth = 2  # Depth of coordinate encoder transformer
    cfg.model.coord_enc_num_heads = 4  # Number of heads in coordinate encoder
    # Weight initialization
    cfg.model.init_weights = "truncnormal"  # Weight initialization method

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.num_points = 10000  # Number of points in the dataset
    cfg.dataset.disk_radius = 1.0  # Radius of the disk
    cfg.dataset.num_control = 20  # Number of control points for deformation
    cfg.dataset.deform_scale = 0.3  # Scale of deformation
    cfg.dataset.kernel_func = "gaussian"  # Type of kernel function for deformation
    cfg.dataset.kernel_epsilon = 0.5  # Epsilon parameter for kernel function

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Training  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.train = ConfigDict()
    cfg.train.batch_size = 128  # Batch size for training
    cfg.train.num_epochs = 100  # Number of training epochs
    cfg.train.lr = 1e-4  # Learning rate
    cfg.train.reg_lambda = 0.1  # Regularization strength
    cfg.train.weight_decay = 1e-3  # Weight decay for optimizer
    cfg.train.reg_lambda_decay = 0.9995  # Decay rate for regularization strength
    cfg.train.reg = "reg+geo"  # Regularization method: "reg", "reg+geo", or "none"
    cfg.train.noise_scale_riemannian = 0.02  # Noise scale for Riemannian metric computation
    cfg.train.lambda_geo_loss = 5.0  # Weight for geodesic loss term
    cfg.train.lambda_g_inv = 0.1  # Weight for inverse metric regularization
    
    return cfg
