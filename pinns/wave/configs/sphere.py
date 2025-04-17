from datetime import datetime
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    config.plot = False

    config.mode = "train"
    config.T = 2.0
    config.c2 = 0.4
    config.sigma_ics = 0.1
    config.amplitude_ics = 50.0

    # Autoencoder checkpoint
    config.autoencoder_checkpoint = ml_collections.ConfigDict()
    config.autoencoder_checkpoint.checkpoint_path = "./fit/checkpoints/sphere"
    config.autoencoder_checkpoint.step = 100

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.log_every_steps = 10
    wandb.eval_every_steps = 100
    wandb.project = "PINN-Debug"
    wandb.name = "default"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 1
    arch.hidden_dim = 32
    arch.out_dim = 1
    arch.activation = "tanh"

    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    # )

    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2, "embed_dim": 128})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.grad_accum_steps = 0
    optim.optimizer = "AdamWarmupCosineDecay"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-5
    optim.decay_rate = 0.9

    optim.lbfgs_learning_rate = 1e-2
    optim.lbfgs_max_steps = 100

    # cosine decay
    optim.warmup_steps = 1000
    optim.decay_steps = 10000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 60000
    training.batch_size = 1024
    training.uniform_sampler_sigma = 0.02

    training.load_existing_batches = True
    
    training.res_batches_path = "pinns/wave/sphere/data/res_batches.npy"
    training.boundary_batches_path = (
        "pinns/wave/sphere/data/boundary_batches.npy"
    )
    training.boundary_pairs_idxs_path = (
        "pinns/wave/sphere/data/boundary_pairs_idxs.npy"
    )
    training.ics_batches_path = "pinns/wave/sphere/data/ics_batches.npy"
    training.ics_values_path = "pinns/wave/sphere/data/ics_values.npy"
    training.ics_derivative_batches_path = "pinns/wave/sphere/data/ics_derivative_batches.npy"
    training.ics_derivative_values_path = "pinns/wave/sphere/data/ics_derivative_values.npy"

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"ics": 0.0, "res": 0.0, "bc": 0.0, "ics_derivative": 0.0}
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 5000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 10000

    config.profiler = profiler = ml_collections.ConfigDict()
    profiler.start_step = 200
    profiler.end_step = 210
    profiler.log_dir = "pinns/wave/sphere/profiler"

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_dir = (
        "pinns/wave/sphere/checkpoints/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 5

    # Eval
    config.eval = eval = ml_collections.ConfigDict()
    eval.eval_with_last_ckpt = True
    eval.checkpoint_dir = "pinns/wave/sphere/checkpoints/"
    eval.step = 11000

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
