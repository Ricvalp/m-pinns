from datetime import datetime

import jax.numpy as jnp
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    config.plot = False

    config.mode = "train"
    config.T = 4.0

    # Autoencoder checkpoint
    config.autoencoder_checkpoint = ml_collections.ConfigDict()
    config.autoencoder_checkpoint.checkpoint_path = "./fit/checkpoints/coil"
    config.autoencoder_checkpoint.step = 60000

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Debug"
    wandb.name = "default"
    wandb.tag = None
    wandb.log_every_steps = 100

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh"

    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    # )

    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.grad_accum_steps = 0
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-5
    optim.lbfgs_learning_rate = 0.00001
    optim.decay_rate = 0.9
    optim.decay_steps = 2000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 500000
    training.batch_size = 1024
    training.lbfgs_max_steps = 0

    training.res_batches_path = (
        "pinns/diffusion_single_gpu_autodecoder/coil/data/res_batches.npy"
    )
    training.boundary_batches_path = (
        "pinns/diffusion_single_gpu_autodecoder/coil/data/boundary_batches.npy"
    )
    training.boundary_pairs_idxs_path = (
        "pinns/diffusion_single_gpu_autodecoder/coil/data/boundary_pairs_idxs.npy"
    )
    training.ics_batches_path = (
        "pinns/diffusion_single_gpu_autodecoder/coil/data/ics_batches.npy"
    )
    training.ics_idxs_path = (
        "pinns/diffusion_single_gpu_autodecoder/coil/data/ics_idxs.npy"
    )

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"ics": 1.0, "res": 1.0, "bc": 1.0}
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 300000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 10000
    logging.log_errors = False
    logging.log_losses = True
    logging.log_weights = False
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    config.profiler = profiler = ml_collections.ConfigDict()
    profiler.start_step = 200
    profiler.end_step = 210
    profiler.log_dir = "pinns/diffusion_single_gpu_autodecoder/coil/profiler"

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_dir = (
        "pinns/diffusion_single_gpu_autodecoder/coil/checkpoints/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 10

    # Eval
    config.eval = eval = ml_collections.ConfigDict()
    eval.eval_with_last_ckpt = True
    eval.checkpoint_dir = "pinns/diffusion_single_gpu_autodecoder/coil/checkpoints/"
    eval.step = 9999

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
