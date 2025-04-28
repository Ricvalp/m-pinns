import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import wandb
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
from jax import vmap
from universal_autoencoder.universal_autoencoder import UniversalAutoencoder
from universal_autoencoder.upt_encoder import EncoderSupernodes
from universal_autoencoder.siren import SirenModel

from chart_autoencoder.utils import ModelCheckpoint, set_profiler


class TrainerUniversalAutoEncoder:

    def __init__(
        self,
        cfg,
        dataset,
        wandb_id,
    ):
        super().__init__()

        self.dataset = dataset

        self.exmp_batch = self.dataset.get_batch()
        # self.exmp_batch = next(iter(self.dataset))[0]

        self.lr = cfg.train.lr
        self.reg_lambda = cfg.train.reg_lambda
        self.reg = cfg.train.reg
        self.wandb_log = cfg.wandb.use
        self.wandb_log_every = cfg.wandb.log_every_steps
        self.save_every = cfg.checkpoint.save_every
        self.rng = jax.random.PRNGKey(cfg.seed)
        self.figure_path = cfg.figure_path
        self.profiler_cfg = cfg.profiler
        self.cfg_model = cfg.model
        self.num_steps = cfg.train.num_steps
        self.lambda_reg_decay = cfg.train.reg_lambda_decay
        self.noise_scale_riemannian = cfg.train.noise_scale_riemannian
        self.lambda_geo_loss = cfg.train.lambda_geo_loss
        self.lambda_g_inv = cfg.train.lambda_g_inv
        self.figure_path = Path(cfg.figure_path)
        self.figure_path.mkdir(parents=True, exist_ok=True)

        self.chart_loader = self.create_data_loader(batch_size=cfg.train.batch_size)

        self.checkpointer = ModelCheckpoint(
            path=Path(cfg.checkpoint.checkpoint_path).absolute() / f"{wandb_id}",
            max_to_keep=1,
            keep_every=1,
            overwrite=cfg.checkpoint.overwrite,
        )

        with open(os.path.join(cfg.checkpoint.checkpoint_path, "cfg.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=4)

        self.rng = jax.random.PRNGKey(cfg.seed)

        self.init_model()
        self.create_functions()


    def create_data_loader(self, batch_size):
        """Create a data loader iterator from the dataset."""
        # Check if dataset has a direct get_batch method
        if hasattr(self.dataset, 'get_batch'):
            # Use generator to create an infinite iterator
            def data_generator():
                while True:
                    yield self.dataset.get_batch()
            return data_generator()
            
        # If dataset supports PyTorch DataLoader interface
        elif hasattr(self.dataset, '__getitem__') and hasattr(self.dataset, '__len__'):
            # Create dataloader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                collate_fn=numpy_collate_with_distances,
            )
            # Create infinite iterator
            def cyclic_iterator():
                while True:
                    for batch in dataloader:
                        yield batch
            return cyclic_iterator()

        else:
            raise ValueError("Dataset must have a get_batch method or support PyTorch DataLoader interface")


    def init_model(self):
        self.states = []
        self.rng, rng = jax.random.split(self.rng)
        
        encoder = EncoderSupernodes(
            input_dim=self.cfg_model.input_dim,
            ndim=self.cfg_model.ndim,
            radius=self.cfg_model.radius,
            max_degree=self.cfg_model.max_degree,
            gnn_dim=self.cfg_model.gnn_dim,
            max_supernodes=self.cfg_model.max_supernodes,
            enc_dim=self.cfg_model.enc_dim,
            enc_depth=self.cfg_model.enc_depth,
            enc_num_heads=self.cfg_model.enc_num_heads,
            perc_dim=self.cfg_model.perc_dim,
            perc_num_heads=self.cfg_model.perc_num_heads,
            num_latent_tokens=self.cfg_model.num_latent_tokens,
            cond_dim=self.cfg_model.cond_dim,
            init_weights=getattr(self.cfg_model, "init_weights", "truncnormal"),
            output_coord_dim=self.cfg_model.output_coord_dim,
            coord_enc_dim=self.cfg_model.coord_enc_dim,
            coord_enc_depth=self.cfg_model.coord_enc_depth,
            coord_enc_num_heads=self.cfg_model.coord_enc_num_heads
        )
        self.encoder_apply_fn = encoder.apply

        decoder = SirenModel(
            coord_dim=self.cfg_model.coord_dim,
            cond_dim=self.cfg_model.cond_dim,
            cond_encoder_features=self.cfg_model.cond_encoder_features,
            siren_features=self.cfg_model.siren_features,
            w0=self.cfg_model.w0,
            w0_initial=self.cfg_model.w0_initial,
        )
        self.decoder_apply_fn = decoder.apply

        model = UniversalAutoencoder(
            coord_dim=self.cfg_model.input_dim,
            radius=self.cfg_model.radius,
            max_degree=self.cfg_model.max_degree,
            gnn_dim=self.cfg_model.gnn_dim,
            max_supernodes=self.cfg_model.max_supernodes,
            enc_dim=self.cfg_model.enc_dim,
            enc_depth=self.cfg_model.enc_depth,
            enc_num_heads=self.cfg_model.enc_num_heads,
            perc_dim=self.cfg_model.perc_dim,
            perc_num_heads=self.cfg_model.perc_num_heads,
            num_latent_tokens=self.cfg_model.num_latent_tokens,
            cond_dim=self.cfg_model.cond_dim,
            cond_encoder_features=self.cfg_model.cond_encoder_features,
            siren_features=self.cfg_model.siren_features,
            w0=self.cfg_model.w0,
            w0_initial=self.cfg_model.w0_initial,
            init_weights=getattr(self.cfg_model, "init_weights", "truncnormal"),
            output_coord_dim=self.cfg_model.output_coord_dim,
            coord_enc_dim=self.cfg_model.coord_enc_dim,
            coord_enc_depth=self.cfg_model.coord_enc_depth,
            coord_enc_num_heads=self.cfg_model.coord_enc_num_heads
        )

        # optimizer = optax.adamw(self.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.001)
        optimizer = optax.adam(self.lr, b1=0.9, b2=0.999, eps=1e-8)
        params = model.init(
            rng,
            self.exmp_batch,
        )["params"]

        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )


    def create_functions(self):

        if self.reg == "reg":

            def loss_fn(params, batch, reg_lambda, key):
                points, distances = batch
                predictions, latent = self.state.apply_fn({"params": params}, points)
                recon_loss = jnp.sum((predictions - points) ** 2, axis=-1).mean()
                riemannian_loss = riemannian_metric_loss(params, points, key)
                return recon_loss + reg_lambda * riemannian_loss, (
                    riemannian_loss,
                    0.0,
                    recon_loss,
                )

        elif self.reg == "reg+geo":

            def loss_fn(params, batch, reg_lambda, key):
                points, distances = batch
                predictions, latent = self.state.apply_fn({"params": params}, points)
                recon_loss = jnp.sum((predictions - points) ** 2, axis=-1).mean()
                riemannian_loss = riemannian_metric_loss(params, points, key)
                geo_loss = geodesic_preservation_loss(distances, latent)
                total_loss = recon_loss + reg_lambda * (
                    riemannian_loss + self.lambda_geo_loss * geo_loss
                )
                return total_loss, (
                    riemannian_loss,
                    geo_loss,
                    recon_loss,
                )

        elif self.reg == "none":

            def loss_fn(params, batch, reg_lambda, key):
                points = batch
                pred = self.state.apply_fn({"params": params}, points)
                recon_loss = jnp.sum((pred - points) ** 2, axis=-1).mean()
                return recon_loss, (0.0, 0.0, recon_loss)

        else:
            raise ValueError(f"Regularization method {self.reg} not defined")

        def geodesic_preservation_loss(distances_matrix, z):
            z_diff = z[:, None, :] - z[None, :, :]
            z_dist = jnp.sqrt(jnp.sum(z_diff**2, axis=-1) + 1e-8)
            z_dist = z_dist / jnp.mean(z_dist)
            geodesic_dist = distances_matrix / jnp.mean(distances_matrix)
            return jnp.mean((z_dist - geodesic_dist) ** 2)

        def riemannian_metric_loss(params, points, key):
            _, z = self.state.apply_fn({"params": params}, points)

            d = lambda z: self.decoder_apply_fn({"params": params["D"]}, z)
            noise = jax.random.normal(key, shape=z.shape) * self.noise_scale_riemannian
            z = z + noise
            J = vmap(jax.jacfwd(d))(z)
            J_T = jnp.transpose(J, (0, 2, 1))
            g = jnp.matmul(J_T, J)
            g_inv = jnp.linalg.inv(g)
            return jnp.mean(jnp.absolute(g)) + self.lambda_g_inv * jnp.mean(
                jnp.absolute(g_inv)
            )

        def train_step(state, batch, reg_lambda, key):
            my_loss = lambda params: loss_fn(params, batch, reg_lambda, key)
            (loss, aux), grads = jax.value_and_grad(my_loss, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss, aux, grads

        self.train_step = jax.jit(train_step)


    def fit(self):
        step = 0
        reg_lambda = self.reg_lambda
    
        progress_bar = tqdm(range(self.num_steps))
        
        for _ in progress_bar:
            try:
                batch = next(self.chart_loader)
                
                self.rng, rng = jax.random.split(self.rng)
                
                self.state, loss, aux, grads = self.train_step(
                    self.state, batch, reg_lambda, rng
                )
                
                reg_lambda = reg_lambda * self.lambda_reg_decay
                
                if self.wandb_log and step % self.wandb_log_every == 0:
                    riemannian_loss, geo_loss, recon_loss = aux
                    
                    log_dict = {
                        "loss": loss,
                        "riemannian_loss": riemannian_loss,
                        "geodesic_loss": geo_loss,
                        "recon_loss": recon_loss,
                        "reg_lambda": reg_lambda,
                        "step": step,
                    }
                        
                    wandb.log(log_dict, step=step)
                    
                if step % self.save_every == 0:
                    self.save_model(step=step)
                
                progress_bar.set_postfix(loss=float(loss), recon=float(aux[2]), reg_lambda=float(reg_lambda))
                
                step += 1
                
            except StopIteration:
                self.chart_loader = self.create_data_loader(batch_size=self.cfg_model.batch_size)
                        
        self.save_model(step=self.num_steps - 1)


    def save_model(self, step):
        self.checkpointer.save_checkpoint(step=step, params=self.state.params)


    def load_model(self, step=None):
        if step is None:
            step = self.checkpointer.get_latest_checkpoint()
        self.state = self.state.replace(
            params=self.checkpointer.load_checkpoint(step=step)
        )


    def model_fn(self, points) -> jnp.ndarray:
        return self.state.apply_fn({"params": self.state.params}, points)


    def decoder_fn(self, z) -> jnp.ndarray:
        return self.decoder_apply_fn({"params": self.state.params["D"]}, z)

