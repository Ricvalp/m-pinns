import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
from jax import vmap
from chart_autoencoder.models import AutoEncoder, Encoder, Decoder
from chart_autoencoder.utils import ModelCheckpoint, set_profiler


class TrainerAutoEncoder:

    def __init__(
        self,
        cfg,
        chart_loader,
        boundary_indices,
        chart,
        chart_3d,
        chart_key,
    ):
        super().__init__()

        self.chart = chart
        self.boundary_indices = boundary_indices
        self.chart_loader = chart_loader
        self.exmp_batch = next(iter(self.chart_loader))[0]
        self.lr = cfg.train.lr
        self.reg_lambda = cfg.train.reg_lambda
        self.reg = cfg.train.reg
        self.wandb_log = cfg.wandb.use
        self.wandb_log_every = cfg.wandb.log_every_steps
        self.save_every = cfg.checkpoint.save_every
        self.log_charts_every = cfg.wandb.log_charts_every
        self.rng = jax.random.PRNGKey(cfg.seed)
        self.figure_path = cfg.figure_path
        self.profiler_cfg = cfg.profiler
        self.cfg_model = cfg.model
        self.num_epochs = cfg.train.num_epochs
        self.chart_3d = chart_3d
        self.chart_key = chart_key
        self.lambda_reg_decay = cfg.train.reg_lambda_decay
        self.noise_scale_riemannian = cfg.train.noise_scale_riemannian
        self.lambda_geo_loss = cfg.train.lambda_geo_loss
        self.lambda_g_inv = cfg.train.lambda_g_inv
        self.figure_path = Path(cfg.figure_path) / f"charts_{self.chart_key}"
        self.figure_path.mkdir(parents=True, exist_ok=True)

        self.checkpointer = ModelCheckpoint(
            path=Path(cfg.checkpoint.checkpoint_path).absolute() / f"chart_{chart_key}",
            max_to_keep=1,
            keep_every=1,
            overwrite=cfg.checkpoint.overwrite,
        )

        with open(os.path.join(cfg.checkpoint.checkpoint_path, "cfg.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=4)

        self.center = jnp.array([[cfg.model.center for _ in range(chart_3d.shape[1])]])

        self.rng = jax.random.PRNGKey(cfg.seed)

        self.init_model()
        self.create_functions()


    def init_model(self):
        self.states = []
        self.rng, rng = jax.random.split(self.rng)
        encoder = Encoder(
            n_hidden=self.cfg_model.n_hidden,
            n_latent=self.cfg_model.n_latent,
            init_scale=self.cfg_model.init_scale,
        )
        self.encoder_apply_fn = encoder.apply

        decoder = Decoder(
            n_hidden=self.cfg_model.n_hidden,
            rff_dim=self.cfg_model.rff_dim,
            n_out=3,
        )
        self.decoder_apply_fn = decoder.apply
        
        model = AutoEncoder(
            n_hidden=self.cfg_model.n_hidden,
            rff_dim=self.cfg_model.rff_dim,
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

            def loss_fn(params, batch, key):
                points, distances = batch
                out, z = self.state.apply_fn({"params": params}, points)
                loss = jnp.sum((out - points) ** 2, axis=-1).mean()
                riemannian_loss = riemannian_metric_loss(params, points, key)
                return jnp.mean(loss) + self.reg_lambda * riemannian_loss, (
                    riemannian_loss,
                    loss,
                )
                
        elif self.reg == "reg+geo":

            def loss_fn(params, batch, reg_lambda, key):
                points, distances = batch
                out, z = self.state.apply_fn({"params": params}, points)
                loss = jnp.sum((out - points) ** 2, axis=-1).mean()
                riemannian_loss = riemannian_metric_loss(params, points, key)
                geo_loss = geodesic_preservation_loss(
                    distances, z
                )
                return loss + reg_lambda * (riemannian_loss + self.lambda_geo_loss * geo_loss), (
                    riemannian_loss,
                    geo_loss,
                    loss,
                )

        elif self.reg == "none":

            def loss_fn(params, batch):
                out = self.state.apply_fn({"params": params})
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                return jnp.mean(loss)

        else:
            raise ValueError(f"Regularization method {self.reg} not defined")

        def geodesic_preservation_loss(distances_matrix, z):
            z_diff = z[:, None, :] - z[None, :, :]
            z_dist = jnp.sqrt(jnp.sum(z_diff**2, axis=-1) + 1e-8)
            z_dist = z_dist / jnp.mean(z_dist)
            geodesic_dist = distances_matrix / jnp.mean(distances_matrix)
            return jnp.mean((z_dist - geodesic_dist) ** 2)

        def riemannian_metric_loss(params, points, key):
            out, z = self.state.apply_fn({"params": params}, points)

            d = lambda z: self.decoder_apply_fn({"params": params["D"]}, z)
            noise = (
                jax.random.normal(key, shape=z.shape)
                * self.noise_scale_riemannian
            )
            z = z + noise
            J = vmap(jax.jacfwd(d))(z)
            J_T = jnp.transpose(J, (0, 2, 1))
            g = jnp.matmul(J_T, J)
            g_inv = jnp.linalg.inv(g)
            return jnp.mean(jnp.absolute(g)) + self.lambda_g_inv * jnp.mean(jnp.absolute(g_inv))

        def train_step(state, batch, reg_lambda, key):
            my_loss = lambda params: loss_fn(params, batch, reg_lambda, key)
            (loss, aux), grads = jax.value_and_grad(my_loss, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss, aux, grads

        self.train_step = jax.jit(train_step)

    def fit(self):

        step = 0
        reg_lambda = self.reg_lambda
        for epoch in range(self.num_epochs):
            for batch in tqdm(self.chart_loader):
                # set_profiler(self.profiler_cfg, step, self.profiler_cfg.log_dir)
                self.rng, rng = jax.random.split(self.rng)
                self.state, loss, aux, grads = self.train_step(
                    self.state, batch, reg_lambda, rng
                )
                reg_lambda = reg_lambda * self.lambda_reg_decay

                if step % self.log_charts_every == 0:

                    x = self.state.apply_fn({"params": self.state.params}, self.chart)[1][:, 0]
                    y = self.state.apply_fn({"params": self.state.params}, self.chart)[1][:, 1]
                    boundaries_x = {}
                    boundaries_y = {}
                    for key in self.boundary_indices.keys():
                        boundary_indices = self.boundary_indices[key]
                        boundaries_x[(key[0], key[1])] = x[jnp.array(boundary_indices)]
                        boundaries_y[(key[0], key[1])] = y[jnp.array(boundary_indices)]
                    fig = plot_domains(
                        x=[x],
                        y=[y],
                        boundaries_x=boundaries_x,
                        boundaries_y=boundaries_y,
                        name=self.figure_path / f"{step}.png",
                    )
                if self.wandb_log and step % self.wandb_log_every == 0:
                    wandb.log(
                        {
                            "loss": loss,
                            # "points grads norm": jnp.linalg.norm(grads["points"]),
                            # "decoder grads norm": jnp.linalg.norm(
                            #     jnp.concatenate(
                            #         [
                            #             jnp.ravel(p)
                            #             for p in jax.tree_util.tree_leaves(grads["D"])
                            #         ]
                            #     )
                            # ),
                            "riemannian_loss": aux[0],
                            "geodesic_loss": aux[1],
                            "recon loss": aux[2],
                            "reg_lambda": reg_lambda,
                            "chart_key": self.chart_key,
                        }, step=step
                    )
                
                step += 1

            # if (step + 1) % self.save_every == 0:
            #     self.save_model(step=step-1)
        self.save_model(step=epoch)

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


class ChartsDataset(Dataset):
    def __init__(self, chart, distance_matrix):
        self.chart = chart
        self.distance_matrix = distance_matrix

    def __len__(self):
        return len(self.chart)

    def __getitem__(self, idx):
        return self.chart[idx], idx, self.distance_matrix[idx]



def product_of_norms(params):
    prod = 1.0
    for key in params.keys():
        prod *= jnp.linalg.norm(params[key]["kernel"])
    return prod


def plot_domains(x, y, boundaries_x, boundaries_y, name=None):
    # Determine the number of plots needed
    num_plots = len(x)
    cols = 4  # You can adjust the number of columns based on your preference
    rows = (num_plots + cols - 1) // cols  # Calculate required rows

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    # Ensure ax is a 2D array for easy indexing
    if num_plots == 1:
        ax = [ax]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    for i in range(num_plots):
        # Calculate row and column index for the plot
        row, col = divmod(i, cols)

        ax[row][col].set_title(f"Chart {i}")
        ax[row][col].scatter(x[i], y[i], s=3, c="b")

        for key in boundaries_x.keys():
            ax[row][col].scatter(
                boundaries_x[key], boundaries_y[key], s=10, label=f"boundary {key}"
            )

        ax[row][col].legend(loc="best")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.close()


def plot_3d_points(x, y, z, title="3D Points"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="r", marker="o")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig
