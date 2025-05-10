import ml_collections
from absl import app
from ml_collections import config_flags
import wandb
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from pathlib import Path
from functools import partial
import jax
import optax
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from flax.training import checkpoints
import json
from universal_autoencoder.upt_autoencoder import UniversalAutoencoder
from universal_autoencoder.siren import ModulatedSIREN
from datasets.universal_ae_dataset import UniversalAEDataset


def load_cfgs():
    """Load configuration from file."""
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.figure_path = "figures/fit_universal_autoencoder_coil"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # Dataset # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = dataset =  ml_collections.ConfigDict()
    dataset.num_supernodes = 32
    dataset.seed = 37
    dataset.create_dataset = False
    dataset.mesh_path = "./datasets/coil/coil_1.2_MM.obj"
    dataset.charts_path = "/scratch-shared/rvalperga/mpinns/datasets/coil/uae_dataset" # "/home/rvalperga/mpinns/datasets/coil/uae_charts"
    dataset.points_per_unit_area = 3
    dataset.subset_cardinality = None
    dataset.num_points = 500
    dataset.iterations = 100
    dataset.min_dist = 10.
    dataset.nearest_neighbors = 10
    dataset.save_charts_every = 10
    dataset.normalize_charts = True

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # Training  # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.train = ml_collections.ConfigDict()
    cfg.train.batch_size = 64
    cfg.train.lr = 1e-4
    cfg.train.num_steps = 500000
    cfg.train.reg = "geodesic_preservation" # "geo+riemannian" # 
    cfg.train.noise_scale_riemannian = 0.01
    cfg.train.num_finetuning_steps = 0
    cfg.train.warmup_lamb_steps = 20000
    cfg.train.max_lamb = 0.0001
    cfg.train.lamb_decay_rate = 0.99995

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # Checkpoint # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.checkpoint = ml_collections.ConfigDict()
    cfg.checkpoint.path = "universal_autoencoder/experiments/coil/checkpoints"
    cfg.checkpoint.save_every = 50000
    cfg.checkpoint.finetuning_save_every = 5000
    cfg.checkpoint.keep = 10
    cfg.checkpoint.overwrite = True

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # Wandb # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.wandb = ml_collections.ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.wandb_log_every = 500
    cfg.wandb.project = "universal_autoencoder-coil"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_riemann_every = 50000
    cfg.wandb.run_name_prefix = "coil"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # Profiler  # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.profiler = ml_collections.ConfigDict()
    cfg.profiler.use = False
    cfg.profiler.log_dir = "universal_autoencoder/profiler/"
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
    cfg.encoder_supernodes_cfg.perc_dim = 256
    cfg.encoder_supernodes_cfg.perc_num_heads = 4
    cfg.encoder_supernodes_cfg.num_latent_tokens = None
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
    cfg.modulated_siren_cfg.num_layers = 2
    cfg.modulated_siren_cfg.hidden_dim = 128
    cfg.modulated_siren_cfg.omega_0 = 1.0
    cfg.modulated_siren_cfg.modulation_hidden_dim = 128
    cfg.modulated_siren_cfg.modulation_num_layers = 3
    cfg.modulated_siren_cfg.shift_modulate = True
    cfg.modulated_siren_cfg.scale_modulate = False

    return cfg


_CONFIG = config_flags.DEFINE_config_dict("config", load_cfgs())


def main(_):
    cfg = _CONFIG.value

    
    if cfg.profiler.use:
        Path(cfg.profiler.log_dir).mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(cfg.seed)
    key, params_subkey, supernode_subkey = jax.random.split(key, 3)

    dataset=UniversalAEDataset(
        config=cfg,
        train=True,
    )
    val_dataset = UniversalAEDataset(
        config=cfg,
        train=False,
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=numpy_collate,
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=numpy_collate,
    )

    run = None
    if cfg.wandb.use:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name_prefix,
            config=cfg.to_dict(),
        )
        wandb_id = run.id
        wandb.run.name = f"{cfg.wandb.run_name_prefix}_{wandb_id}"
        wandb.run.save()
    else:
        wandb_id = "no_wandb_" + str(cfg.seed)
    
    figure_path = cfg.figure_path + f"/{wandb_id}"
    Path(figure_path).mkdir(parents=True, exist_ok=True)

    (Path(cfg.checkpoint.path) / wandb_id).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cfg.checkpoint.path, f"{wandb_id}/cfg.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=4)

    model = UniversalAutoencoder(cfg=cfg)
    decoder = ModulatedSIREN(cfg=cfg)
    decoder_apply_fn = decoder.apply

    @partial(jax.vmap, in_axes=(0, 0))
    def geodesic_preservation_loss(distances_matrix, z):
        z_diff = z[:, None, :] - z[None, :, :]
        z_dist = jnp.sqrt(jnp.sum(z_diff**2, axis=-1) + 1e-8)
        z_dist = z_dist / jnp.mean(z_dist)
        geodesic_dist = distances_matrix / jnp.mean(distances_matrix)
        return jnp.mean((z_dist - geodesic_dist) ** 2)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    def riemannian_metric_loss(params, latent, coords):
        d = lambda x: decoder_apply_fn(
            {"params": params["siren"]}, x, latent
        )
        J = jax.vmap(jax.jacfwd(d))(coords)[:, 0, :, :]
        J_T = jnp.transpose(J, (0, 2, 1))
        g = jnp.matmul(J_T, J)
        g_inv = jnp.linalg.inv(g)
        return jnp.mean(jnp.absolute(g)) + 0.1 * jnp.mean(jnp.absolute(g_inv))

    distance_matrix = jnp.array(dataset.distance_matrix)

    exmp_chart, exmp_supernode_idxs, exmp_chart_id = next(iter(data_loader))
    plot_dataset(exmp_chart, exmp_supernode_idxs, distance_matrix[exmp_chart_id], name=figure_path + "/coil_dataset_with_supernodes.png")
    val_exmp_chart, val_exmp_supernode_idxs, val_exmp_chart_id = next(iter(val_data_loader))
    plot_dataset(val_exmp_chart, val_exmp_supernode_idxs, distance_matrix[val_exmp_chart_id], name=figure_path + "/coil_dataset_with_supernodes_val.png")

    def geo_riemann_loss_fn(params, batch, key, lamb=1.0):

        points, supernode_idxs, chart_id = batch
        pred, coords, conditioning = state.apply_fn({"params": params}, points, supernode_idxs)
        recon_loss = jnp.sum((pred - points) ** 2, axis=-1).mean()

        noise = (
            jax.random.normal(key, shape=coords.shape)
            * cfg.train.noise_scale_riemannian
        )

        geodesic_loss = geodesic_preservation_loss(distance_matrix[chart_id], coords).mean()
        riemannian_loss = riemannian_metric_loss(params, conditioning, coords + noise).mean()

        return recon_loss + lamb * (geodesic_loss + riemannian_loss), (recon_loss, geodesic_loss, riemannian_loss)


    def geo_loss_fn(params, batch, key, lamb=3.0):
        points, supernode_idxs, chart_id = batch
        pred, coords, conditioning = state.apply_fn({"params": params}, points, supernode_idxs)
        recon_loss = jnp.sum((pred - points) ** 2, axis=-1).mean()
        geodesic_loss = geodesic_preservation_loss(distance_matrix[chart_id], coords).mean()
        return recon_loss + lamb * geodesic_loss, (recon_loss, geodesic_loss)


    @jax.jit
    def train_step_riemann(state, batch, key, lamb):
        my_loss = lambda params: geo_riemann_loss_fn(params, batch, key, lamb)
        (loss, aux), grads = jax.value_and_grad(my_loss, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux, grads

    @jax.jit
    def train_step(state, batch, key):
        my_loss = lambda params: geo_loss_fn(params, batch, key)
        (loss, aux), grads = jax.value_and_grad(my_loss, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux, grads

    batch_size = cfg.train.batch_size
    num_points = cfg.dataset.num_points
    num_supernodes = cfg.encoder_supernodes_cfg.max_degree

    # Optional condition
    supernode_idxs = jax.random.randint(
        supernode_subkey, (batch_size, num_supernodes), 0, num_points
    )

    # Initialize parameters
    init_points, supernode_idxs, _ = next(iter(data_loader))
    params = model.init(params_subkey, init_points, supernode_idxs)["params"]
    print(f"Number of parameters: {count_parameters(params)}")
    optimizer = optax.adam(learning_rate=cfg.train.lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

# ------------------------------
# ------- training loop --------
# ------------------------------

    progress_bar = tqdm(range(cfg.train.num_steps))
    step = 1
    data_loader_iter = iter(data_loader)

    geodesic_riemann_loss_fn = jax.jit(geo_riemann_loss_fn)

    for _ in progress_bar:

        if cfg.profiler.use:
            set_profiler(cfg.profiler, step, cfg.profiler.log_dir)

        try:
            batch = next(data_loader_iter)
            key, subkey = jax.random.split(key)
            state, loss, aux, grads = train_step(state, batch, subkey)

            if step % cfg.wandb.wandb_log_every == 0:

                if step % cfg.wandb.log_riemann_every == 0:
                    val_batch = next(iter(val_data_loader)) 
                    loss_riemann, aux_riemann = geodesic_riemann_loss_fn(state.params, val_batch, subkey, lamb=0.0)
                    wandb.log({"riemannian_loss": aux_riemann[2]}, step=step)
                    name = figure_path + f"/reconstruction_samples_riemannian_loss_{step}.png"
                    test_reconstruction(state, val_data_loader, decoder_apply_fn, name=name)
                    wandb.log({"reconstruction_samples": wandb.Image(name)}, step=step)
                    wandb.log({
                        "val_recon_loss": aux_riemann[0],
                        "val_geodesic_loss": aux_riemann[1],
                        "val_riemannian_loss": aux_riemann[2],
                    }, step=step)

                if cfg.train.reg == "geo+riemannian":
                    log_dict = {
                        "loss": loss,
                        "recon_loss": aux[0],
                        "geodesic_loss": aux[1],
                        "riemannian_loss": aux[2],
                    }
                elif cfg.train.reg == "geodesic_preservation":
                    log_dict = {
                        "loss": loss,
                        "recon_loss": aux[0],
                        "geodesic_loss": aux[1],
                    }
                elif cfg.train.reg == "none":
                    log_dict = {
                        "loss": loss,
                    }

                if cfg.wandb.use:
                    wandb.log(log_dict, step=step)

            if step % cfg.checkpoint.save_every == 0:
                save_checkpoint(state, cfg.checkpoint.path + f"/{wandb_id}", keep=cfg.checkpoint.keep, overwrite=cfg.checkpoint.overwrite)

            progress_bar.set_postfix(loss=float(loss))

            step += 1

        except StopIteration:
            
            data_loader_iter = iter(data_loader)
    
    # save_checkpoint(state, cfg.checkpoint.path + f"/{wandb_id}", keep=cfg.checkpoint.keep, overwrite=cfg.checkpoint.overwrite)

# ------------------------------
# --------- testing ------------
# ------------------------------

    # Add test reconstruction after training
    print("Testing reconstruction...")
    name = figure_path + "/final_reconstruction_samples_pre_finetuning.png"
    test_mse = test_reconstruction(state, data_loader, decoder_apply_fn, name=name)

    if cfg.wandb.use:
        wandb.log({"final_reconstruction_mse": test_mse})
        wandb.log({"reconstruction_samples": wandb.Image(name)})

# ------------------------------
# --------- finetuning --------- 
# ------------------------------

    progress_bar = tqdm(range(cfg.train.num_finetuning_steps))
    global_step = step
    step = 1
    data_loader_iter = iter(data_loader)

    lamb = 0.0001

    for _ in progress_bar:

        if step < cfg.train.warmup_lamb_steps:
            lamb = cfg.train.max_lamb * (step/cfg.train.warmup_lamb_steps)
        else:
            lamb = lamb * cfg.train.lamb_decay_rate

        try:
            batch = next(data_loader_iter)
            key, subkey = jax.random.split(key)
            state, loss, aux, grads = train_step_riemann(state, batch, subkey, lamb=lamb)

            if step % cfg.wandb.wandb_log_every == 0 and cfg.wandb.use:

                log_dict = {
                    "loss": loss,
                    "recon_loss": aux[0],
                    "geodesic_loss": aux[1],
                    "riemannian_loss": aux[2],
                    "lamb": lamb,
                }

                wandb.log(log_dict, step=global_step)

            if global_step % cfg.checkpoint.finetuning_save_every == 0:
                save_checkpoint(state, cfg.checkpoint.path + f"/{wandb_id}", keep=cfg.checkpoint.keep, overwrite=cfg.checkpoint.overwrite)

            progress_bar.set_postfix(loss=float(loss))

            step += 1
            global_step += 1

        except StopIteration:
            
            data_loader_iter = iter(data_loader)
    
    save_checkpoint(state, cfg.checkpoint.path + f"/{wandb_id}", keep=cfg.checkpoint.keep, overwrite=cfg.checkpoint.overwrite)

# ------------------------------
# --- testing post finetuning --
# ------------------------------

    print("Testing reconstruction...")
    name = figure_path + "/reconstruction_samples_post_finetuning.png"
    test_mse = test_reconstruction(state, data_loader, decoder_apply_fn, name=name)

    if cfg.wandb.use:
        wandb.log({"final_reconstruction_mse": test_mse})
        wandb.log({"reconstruction_samples": wandb.Image(name)})

# ------------------------------
# ------ getting charts --------
# ------------------------------

    # coords = jnp.concatenate(coords, axis=0)

    # with open(f"./datasets/{dataset}/charts/charts2d.pkl", "wb") as f:
    #     pickle.dump(coords, f)


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


def test_reconstruction(state, data_loader, decoder_apply_fn, num_samples=5, name="reconstruction_samples"):
    """Test the reconstruction ability of the model and visualize results in 3D.
    
    Args:
        state: The trained model state
        data_loader: DataLoader to get test samples
        num_samples: Number of samples to visualize
    """
    # Get a batch of data
    data_iter = iter(data_loader)
    points, supernode_idxs, chart_id = next(data_iter)
    
    # Generate reconstructions
    reconstructions, coords, conditioning = state.apply_fn({"params": state.params}, points, supernode_idxs)

    # Calculate Riemannian metric determinant
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def riemannian_metric_norm(params, latent, coords):
        d = lambda x: decoder_apply_fn(
            {"params": params["siren"]}, x, latent
        )
        J = jax.vmap(jax.jacfwd(d))(coords)[:, 0, :, :]
        J_T = jnp.transpose(J, (0, 2, 1))
        g = jnp.matmul(J_T, J)
        g_inv = jnp.linalg.inv(g)
        return jnp.linalg.norm(g, axis=(1, 2)), jnp.linalg.norm(g_inv, axis=(1, 2))

    det_g, det_g_inv = riemannian_metric_norm(state.params, conditioning, coords)

    # Convert to numpy for plotting
    points_np = np.array(points)
    reconstructions_np = np.array(reconstructions)
    coords_np = np.array(coords)
    det_g_np = np.array(det_g)
    det_g_inv_np = np.array(det_g_inv)
    
    # Compute global ranges for 3D plots (across all samples)
    # For original points
    x_min_orig = points_np[:, :, 0].min()
    x_max_orig = points_np[:, :, 0].max()
    y_min_orig = points_np[:, :, 1].min()
    y_max_orig = points_np[:, :, 1].max()
    z_min_orig = points_np[:, :, 2].min()
    z_max_orig = points_np[:, :, 2].max()
    
    # For reconstructed points
    x_min_recon = reconstructions_np[:, :, 0].min()
    x_max_recon = reconstructions_np[:, :, 0].max()
    y_min_recon = reconstructions_np[:, :, 1].min()
    y_max_recon = reconstructions_np[:, :, 1].max()
    z_min_recon = reconstructions_np[:, :, 2].min()
    z_max_recon = reconstructions_np[:, :, 2].max()
    
    # Get the min/max across both sets of points for consistent scaling
    x_min_3d = min(x_min_orig, x_min_recon)
    x_max_3d = max(x_max_orig, x_max_recon)
    y_min_3d = min(y_min_orig, y_min_recon)
    y_max_3d = max(y_max_orig, y_max_recon)
    z_min_3d = min(z_min_orig, z_min_recon)
    z_max_3d = max(z_max_orig, z_max_recon)
    
    # Add 10% padding
    x_pad_3d = 0.1 * (x_max_3d - x_min_3d)
    y_pad_3d = 0.1 * (y_max_3d - y_min_3d)
    z_pad_3d = 0.1 * (z_max_3d - z_min_3d)
    
    # Set the global limits for 3D plots
    x_limits_3d = [x_min_3d - x_pad_3d, x_max_3d + x_pad_3d]
    y_limits_3d = [y_min_3d - y_pad_3d, y_max_3d + y_pad_3d]
    z_limits_3d = [z_min_3d - z_pad_3d, z_max_3d + z_pad_3d]
    
    # Compute global ranges for 2D coordinate plots (across all samples)
    x_min_2d = coords_np[:, :, 0].min()
    x_max_2d = coords_np[:, :, 0].max()
    y_min_2d = coords_np[:, :, 1].min()
    y_max_2d = coords_np[:, :, 1].max()
    
    # Add 10% padding for 2D plots
    x_pad_2d = 0.1 * (x_max_2d - x_min_2d)
    y_pad_2d = 0.1 * (y_max_2d - y_min_2d)
    
    # Set the global limits for 2D plots
    x_limits_2d = [x_min_2d - x_pad_2d, x_max_2d + x_pad_2d]
    y_limits_2d = [y_min_2d - y_pad_2d, y_max_2d + y_pad_2d]
    
    # Global color map ranges for the metric values
    vmin_g = det_g_np.min()
    vmax_g = det_g_np.max()
    vmin_g_inv = det_g_inv_np.min()
    vmax_g_inv = det_g_inv_np.max()
    
    # Visualize the first num_samples examples
    fig = plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(min(num_samples, len(points_np))):
        # Original points
        ax1 = fig.add_subplot(num_samples, 4, 4*i+1, projection='3d')
        ax1.scatter(
            points_np[i, :, 0], 
            points_np[i, :, 1], 
            points_np[i, :, 2], 
            c='blue', alpha=0.6, s=10
        )
        if i == 0:
            ax1.set_title(f'Original Points')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Use global limits for 3D plots
        ax1.set_xlim(x_limits_3d)
        ax1.set_ylim(y_limits_3d)
        ax1.set_zlim(z_limits_3d)
        
        # Reconstructed points
        ax2 = fig.add_subplot(num_samples, 4, 4*i+2, projection='3d')
        ax2.scatter(
            reconstructions_np[i, :, 0], 
            reconstructions_np[i, :, 1], 
            reconstructions_np[i, :, 2], 
            c='red', alpha=0.6, s=10
        )
        if i == 0:
            ax2.set_title(f'Reconstructed Points')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Use the same global limits for 3D plots
        ax2.set_xlim(x_limits_3d)
        ax2.set_ylim(y_limits_3d)
        ax2.set_zlim(z_limits_3d)

        # Learned coordinates
        ax3 = fig.add_subplot(num_samples, 4, 4*i+3)
        scatter = ax3.scatter(
            coords_np[i, :, 0], 
            coords_np[i, :, 1],
            c=det_g_inv_np[i],
            alpha=0.8,
            s=10,
            vmin=vmin_g_inv,
            vmax=vmax_g_inv
        )
        if i == 0:
            ax3.set_title(f'Learned 2D Coordinates')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_xlim(x_limits_2d)
        ax3.set_ylim(y_limits_2d)
        ax3.set_aspect('equal')
        plt.colorbar(scatter, ax=ax3, label=r'$||g^{-1}||$')

        ax4 = fig.add_subplot(num_samples, 4, 4*i+4)
        scatter = ax4.scatter(
            coords_np[i, :, 0], 
            coords_np[i, :, 1],
            c=det_g_np[i],
            alpha=0.8,
            s=10,
            vmin=vmin_g,
            vmax=vmax_g
        )
        if i == 0:
            ax4.set_title(f'g')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_xlim(x_limits_2d)
        ax4.set_ylim(y_limits_2d)
        ax4.set_aspect('equal')
        plt.colorbar(scatter, ax=ax4, label=r'$||g||$')
    
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, dpi=300)
    plt.close()
    
    # Calculate reconstruction error
    mse = np.mean(np.sum((points_np - reconstructions_np) ** 2, axis=-1))
    print(f"Reconstruction MSE: {mse:.6f}")
    
    return mse


def save_checkpoint(state, path, keep=5, overwrite=False):
    if not os.path.isdir(path):
        os.makedirs(path)

    step = int(state.step)
    checkpoints.save_checkpoint(Path(path).absolute(), state, step=step, keep=keep, overwrite=overwrite)


def plot_dataset(chart, supernode_idxs, distance_matrix, name=None):
    """
    Plot a batch of charts with the supernodes highlighted in a different color.
    
    Args:
        chart: Batch of 3D points representing charts
        supernode_idxs: Indices of supernodes for each chart
        distance_matrix: Distance matrix of the charts
        name: Optional path to save the figure
    """
    # Determine grid layout
    num_samples = min(16, len(chart))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    # Compute global min/max across all charts for consistent scaling
    x_min_global = np.min([chart[i][:, 0].min() for i in range(num_samples)])
    x_max_global = np.max([chart[i][:, 0].max() for i in range(num_samples)])
    y_min_global = np.min([chart[i][:, 1].min() for i in range(num_samples)])
    y_max_global = np.max([chart[i][:, 1].max() for i in range(num_samples)])
    z_min_global = np.min([chart[i][:, 2].min() for i in range(num_samples)])
    z_max_global = np.max([chart[i][:, 2].max() for i in range(num_samples)])
    
    # Add 10% padding to the global ranges
    x_pad = 0.1 * (x_max_global - x_min_global)
    y_pad = 0.1 * (y_max_global - y_min_global)
    z_pad = 0.1 * (z_max_global - z_min_global)
    
    # Compute global limits
    x_limits = [x_min_global - x_pad, x_max_global + x_pad]
    y_limits = [y_min_global - y_pad, y_max_global + y_pad]
    z_limits = [z_min_global - z_pad, z_max_global + z_pad]
    
    # Find global min/max for colormap scaling (distances)
    all_distances = np.concatenate([distance_matrix[i][0] for i in range(num_samples)])
    distance_min = all_distances.min()
    distance_max = all_distances.max()
    
    fig = plt.figure(figsize=(12, num_samples))
    
    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Create a mask for supernodes
        is_supernode = np.zeros(len(chart[i]), dtype=bool)
        is_supernode[supernode_idxs[i]] = True

        distances = distance_matrix[i]
        
        # Plot regular points with consistent colormap scaling
        ax.scatter(
            chart[i][~is_supernode, 0],
            chart[i][~is_supernode, 1],
            chart[i][~is_supernode, 2],
            c=distances[0][~is_supernode],
            alpha=0.5,
            s=10,
            label='Regular Points',
            vmin=distance_min,
            vmax=distance_max
        )
        
        # Plot supernodes with different color
        ax.scatter(
            chart[i][is_supernode, 0],
            chart[i][is_supernode, 1],
            chart[i][is_supernode, 2],
            c='red',
            alpha=1.0,
            s=30,
            label='Supernodes'
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Use global limits for all plots
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.legend()
    
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, dpi=300)
    plt.close()


if __name__ == "__main__":
    app.run(main)