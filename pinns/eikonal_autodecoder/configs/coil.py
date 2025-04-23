from datetime import datetime

import jax.numpy as jnp
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Plotting the charts at the beginning of the training
    config.plot = False

    # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    config.mode = "train"
    config.N = 100
    config.idxs = [3461, 4175, 4865, 5338, 5731, 6239, 1333, 6886, 7580, 3094, 8521, 2245, 9640, 9831, 2678, 11078, 11210]

    # Autoencoder checkpoint
    config.autoencoder_checkpoint = ml_collections.ConfigDict()
    config.autoencoder_checkpoint.checkpoint_path = "./fit/checkpoints/coil"
    config.autoencoder_checkpoint.step = 60000

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Eikonal-Coil"
    wandb.name = "default"
    wandb.tag = None
    wandb.entity = "ricvalp"

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 1
    arch.hidden_dim = 16
    arch.out_dim = 1
    arch.activation = "tanh"

    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    # )

    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 32})
    # arch.reparam = ml_collections.ConfigDict(
    #     {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    # )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.grad_accum_steps = 0
    optim.optimizer ="Adam" # "AdamWarmupCosineDecay"  #
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.lbfgs_learning_rate = 0.00001
    optim.decay_rate = 0.9
    optim.decay_steps = 2000

    # cosine decay
    optim.warmup_steps = 5000
    optim.decay_steps = 50000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 20000
    training.batch_size = 1024
    training.lbfgs_max_steps = 0

    training.load_existing_batches = True
    training.batches_path = "pinns/eikonal_autodecoder/coil/data/"

    training.res_batches_path = "pinns/eikonal_autodecoder/coil/data/res_batches.npy"
    training.boundary_batches_path = (
        "pinns/eikonal_autodecoder/coil/data/boundary_batches.npy"
    )
    training.boundary_pairs_idxs_path = (
        "pinns/eikonal_autodecoder/coil/data/boundary_pairs_idxs.npy"
    )
    training.bcs_batches_path = "pinns/eikonal_autodecoder/coil/data/bcs_batches.npy"
    training.bcs_values_path = "pinns/eikonal_autodecoder/coil/data/bcs_values.npy"


    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"bcs": 1.0, "res": 1.0, "bc": 1.0}
    )
    weighting.momentum = 0.99
    weighting.update_every_steps = 100

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.num_eval_points = 10000
    logging.log_every_steps = 100
    logging.eval_every_steps = 100

    # config.profiler = profiler = ml_collections.ConfigDict()
    # profiler.start_step = 200
    # profiler.end_step = 210
    # profiler.log_dir = "pinns/eikonal_autodecoder/coil/profiler"

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_dir = (
        "pinns/eikonal_autodecoder/coil/checkpoints/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    saving.save_every_steps = 50000
    saving.num_keep_ckpts = 2
    saving.csv_path = "pinns/eikonal_autodecoder/coil/ablation.csv"

    # Eval
    config.eval = eval = ml_collections.ConfigDict()
    eval.eval_with_last_ckpt = False
    eval.checkpoint_dir = "pinns/eikonal_autodecoder/coil/checkpoints/"
    eval.step = 9999
    eval.N = 11769
    eval.use_existing_solution = False
    eval.solution_path = "pinns/eikonal_autodecoder/coil/eval"

    eval.plot_everything = False

    return config
