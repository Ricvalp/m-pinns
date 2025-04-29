from absl import app, logging
import ml_collections
import wandb
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from torch.utils.data import Dataset
from pathlib import Path
import jax
import optax
from tqdm import tqdm
from torch.utils.data import DataLoader

from universal_autoencoder.upt_autoencoder import UniversalAutoencoder


def load_cfgs():
    """Load configuration from file."""
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.figure_path = "figures/test_fit_universal_autoencoder"

    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.num_points = 1000
    cfg.dataset.disk_radius = 1.0
    cfg.dataset.num_supernodes = 32

    cfg.train = ml_collections.ConfigDict()
    cfg.train.batch_size = 4
    cfg.train.lr = 1e-3
    cfg.train.num_steps = 10000

    cfg.wandb = ml_collections.ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.wandb_log_every = 1

    cfg.profiler = ml_collections.ConfigDict()
    cfg.profiler.log_dir = "test_universal_autoencoder/profiler/"
    cfg.profiler.start_step = 20
    cfg.profiler.end_step = 30

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # #   EncoderSupernodes # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.encoder_supernodes_cfg = ml_collections.ConfigDict()
    cfg.encoder_supernodes_cfg.max_degree = 5
    cfg.encoder_supernodes_cfg.input_dim = 3
    cfg.encoder_supernodes_cfg.gnn_dim = 64
    cfg.encoder_supernodes_cfg.enc_dim = 64
    cfg.encoder_supernodes_cfg.enc_depth = 4
    cfg.encoder_supernodes_cfg.enc_num_heads = 4
    cfg.encoder_supernodes_cfg.perc_dim = 64
    cfg.encoder_supernodes_cfg.perc_num_heads = 4
    cfg.encoder_supernodes_cfg.num_latent_tokens = 8
    cfg.encoder_supernodes_cfg.init_weights = "truncnormal"
    cfg.encoder_supernodes_cfg.output_coord_dim = 2
    cfg.encoder_supernodes_cfg.coord_enc_dim = 64
    cfg.encoder_supernodes_cfg.coord_enc_depth = 2
    cfg.encoder_supernodes_cfg.coord_enc_num_heads = 4
    cfg.encoder_supernodes_cfg.latent_encoder_depth = 2
    cfg.encoder_supernodes_cfg.ndim = 3
    cfg.encoder_supernodes_cfg.perc_depth = 4

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  SIREN  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.modulated_siren_cfg = ml_collections.ConfigDict()
    cfg.modulated_siren_cfg.output_dim = 3
    cfg.modulated_siren_cfg.num_layers = 4
    cfg.modulated_siren_cfg.hidden_dim = 256
    cfg.modulated_siren_cfg.omega_0 = 30.0
    cfg.modulated_siren_cfg.modulation_hidden_dim = 256
    cfg.modulated_siren_cfg.modulation_num_layers = 4
    cfg.modulated_siren_cfg.shift_modulate = True
    cfg.modulated_siren_cfg.scale_modulate = False

    return cfg


def main(_):
    cfg = load_cfgs()

    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.profiler.log_dir).mkdir(parents=True, exist_ok=True)
    
    key = jax.random.PRNGKey(cfg.seed)
    key, params_subkey, supernode_subkey = jax.random.split(key, 3)
    data_loader = DataLoader(
        dataset=HalfSphereDataset(
            num_points=cfg.dataset.num_points, num_supernodes=cfg.dataset.num_supernodes
        ),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=numpy_collate,
    )

    run = None
    if cfg.wandb.use:
        run = wandb.init(
            project="test_universal_autoencoder",
            entity="ricvalp",
            name=f"test_universal_autoencoder",
            config=cfg.to_dict(),
        )
        wandb_id = run.id
    else:
        wandb_id = "no_wandb_" + str(cfg.seed)

    def loss_fn(params, batch):
        points, supernode_idxs = batch
        pred = state.apply_fn({"params": params}, points, supernode_idxs)
        recon_loss = jnp.sum((pred - points) ** 2, axis=-1).mean()
        return recon_loss

    def train_step(state, batch):
        my_loss = lambda params: loss_fn(params, batch)
        loss, grads = jax.value_and_grad(my_loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, grads

    batch_size = cfg.train.batch_size
    num_points = cfg.dataset.num_points
    num_supernodes = cfg.encoder_supernodes_cfg.max_degree

    # Optional condition
    supernode_idxs = jax.random.randint(
        supernode_subkey, (batch_size, num_supernodes), 0, num_points
    )
    # Initialize model
    model = UniversalAutoencoder(cfg=cfg)

    # Initialize parameters
    init_points, supernode_idxs = next(iter(data_loader))
    params = model.init(params_subkey, init_points, supernode_idxs)["params"]
    print(f"Number of parameters: {count_parameters(params)}")
    optimizer = optax.adam(learning_rate=cfg.train.lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    progress_bar = tqdm(range(cfg.train.num_steps))
    step = 0
    data_loader_iter = iter(data_loader)

    for _ in progress_bar:

        set_profiler(cfg.profiler, step, cfg.profiler.log_dir)

        batch = next(data_loader_iter)

        state, loss, grads = train_step(state, batch)

        if step % cfg.wandb.wandb_log_every == 0:

            log_dict = {
                "loss": loss,
                "step": step,
            }
            if cfg.wandb.use:
                wandb.log(log_dict, step=step)

        progress_bar.set_postfix(loss=float(loss))

        step += 1


class HalfSphereDataset(Dataset):
    def __init__(self, num_points, num_supernodes):
        self.num_points = num_points
        self.num_supernodes = num_supernodes
        theta = np.random.uniform(0, jnp.pi, (num_points, 1))
        phi = np.random.uniform(0, 2 * jnp.pi, (num_points, 1))
        self.points = np.concatenate(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        )

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        supernode_idxs = np.random.permutation(self.num_points)[: self.num_supernodes]
        return self.points, supernode_idxs


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def set_profiler(profiler_config, step, log_dir):
    if profiler_config is not None:
        if step == profiler_config.start_step:
            jax.profiler.start_trace(log_dir=log_dir)
        if step == profiler_config.end_step:
            jax.profiler.stop_trace()


def count_parameters(params):
    """Count the number of parameters in a parameter tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


if __name__ == "__main__":
    app.run(main)