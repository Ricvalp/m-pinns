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
from mpl_toolkits.mplot3d import Axes3D

from chart_autoencoder.utils import ModelCheckpoint, get_model, set_profiler


class Trainer:

    def __init__(
        self,
        cfg,
        data_loader,
        charts,
        boundaries,
    ):
        super().__init__()

        self.lr = cfg.train.lr
        self.reg_lambda = cfg.train.reg_lambda
        self.reg = cfg.train.reg
        self.wandb_log = cfg.wandb.use
        self.save_every = cfg.checkpoint.save_every
        self.log_charts_every = cfg.wandb.log_charts_every
        self.rng = jax.random.PRNGKey(cfg.seed)
        self.figure_path = cfg.figure_path

        self.profiler_cfg = cfg.profiler

        self.model = get_model(cfg.model)

        self.checkpointer = ModelCheckpoint(
            path=Path(cfg.checkpoint.checkpoint_path).absolute(),
            max_to_keep=1,
            keep_every=1,
            overwrite=cfg.checkpoint.overwrite,
        )

        with open(os.path.join(cfg.checkpoint.checkpoint_path, "cfg.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=4)

        self.data_loader = data_loader
        self.example_input = next(data_loader)[0]
        self.n_charts = self.example_input.shape[0]
        self.center = jnp.array([[cfg.model.center for _ in range(2)]])
        self.num_steps = cfg.train.num_steps
        self.charts = charts
        self.boundaries = boundaries

        self.create_functions()

    def init_model(self):

        # lr_schedule = optax.warmup_cosine_decay_schedule(
        #     init_value=0.0,
        #     peak_value=self.lr,
        #     warmup_steps=100,
        #     decay_steps=num_steps,
        #     end_value=self.lr / 100,
        # )
        # optimizer = optax.adam(lr_schedule)

        optimizer = optax.adamw(self.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.001)

        # optimizer = optax.adam(self.lr)

        vmapped_init = jax.vmap(self.model.init, in_axes=(0, None))
        params = vmapped_init(
            jax.random.split(self.rng, self.n_charts), self.example_input
        )["params"]

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
        )

    def create_functions(self):

        if self.reg == "reg":

            def loss_fn(params, batch):
                batch, batch_indices = batch
                out, z = self.model.apply({"params": params}, batch)
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                reg = jnp.sum((z - self.center) ** 2, axis=-1)
                return jnp.mean(loss) + self.reg_lambda * jnp.mean(reg)

        elif self.reg == "reg+iso":

            def loss_fn(params, batch):
                batch, _ = batch
                out, z = self.model.apply({"params": params}, batch)
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                reg = jnp.sum((z - self.center) ** 2, axis=-1)
                iso_loss = isometry_loss(batch, z)
                return jnp.mean(loss) + self.reg_lambda * (jnp.mean(reg) + iso_loss)

        elif self.reg == "reg+geo":

            def loss_fn(params, batch):
                batch, distances_matrix = batch
                out, z = self.model.apply({"params": params}, batch)
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                reg = jnp.sum((z - self.center) ** 2, axis=-1)
                geo_loss = geodesic_preservation_loss(distances_matrix, z)
                return jnp.mean(loss) + self.reg_lambda * (jnp.mean(reg) + 5 * geo_loss)

        elif self.reg == "reg+iso+geo":

            def loss_fn(params, batch):
                batch, distances_matrix = batch
                out, z = self.model.apply({"params": params}, batch)
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                reg = jnp.sum((z - self.center) ** 2, axis=-1)
                iso_loss = isometry_loss(batch, z)
                geo_loss = geodesic_preservation_loss(distances_matrix, z)
                return jnp.mean(loss) + self.reg_lambda * (
                    jnp.mean(reg) + iso_loss + 10 * geo_loss
                )

        elif self.reg == "none":

            def loss_fn(params, batch):
                batch, _ = batch
                out, z = self.model.apply({"params": params}, batch)
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                return jnp.mean(loss)

        # def regularizer_fn(params):
        #     prod_decoder = jnp.array(
        #         [product_of_norms(params[f"D"]) for i in range(self.n_charts)]
        #     )
        #     max_decoder = jnp.max(prod_decoder)
        #     sum_decoders = jnp.sum(prod_decoder)
        #     return max_decoder + 0.1 * sum_decoders

        def geodesic_preservation_loss(distances_matrix, z):
            # Compute pairwise Euclidean distances in latent space
            z_diff = z[:, None, :] - z[None, :, :]
            z_dist = jnp.sqrt(jnp.sum(z_diff**2, axis=-1) + 1e-8)
            z_dist = z_dist / jnp.mean(z_dist)
            batch_geodesic_dist = distances_matrix / jnp.mean(distances_matrix)
            return jnp.mean((z_dist - batch_geodesic_dist) ** 2)

        def isometry_loss(batch, z):
            diff = batch[:, None, :] - batch[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
            z_diff = z[:, None, :] - z[None, :, :]
            z_dist = jnp.sqrt(jnp.sum(z_diff**2, axis=-1) + 1e-8)
            return jnp.mean((dist - z_dist) ** 2)

        def train_step(state, batch):
            my_loss = lambda params: jax.vmap(loss_fn, in_axes=(0, 0))(
                params, batch
            ).mean()
            loss, grads = jax.value_and_grad(my_loss, has_aux=False)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        self.train_step = jax.jit(train_step)  # train_step #

    def fit(self, num_steps=1000):

        self.init_model()

        with tqdm(total=100) as pbar:
            step = 0
            progress = 0
            for batch in self.data_loader:

                set_profiler(self.profiler_cfg, step, self.profiler_cfg.log_dir)
                self.state, loss = self.train_step(self.state, batch)
                if self.wandb_log:
                    wandb.log({"loss": loss})

                    if step % self.log_charts_every == 0:
                        x = [
                            self.encoder_fn(self.charts[i], i)[:, 0]
                            for i in range(self.n_charts)
                        ]
                        y = [
                            self.encoder_fn(self.charts[i], i)[:, 1]
                            for i in range(self.n_charts)
                        ]
                        boundaries_x = {}
                        boundaries_y = {}
                        for key in self.boundaries.keys():
                            if len(key) == 2:
                                boundary = self.boundaries[key]
                                boundaries_x[(key[0], key[1])] = self.encoder_fn(
                                    boundary, key[0]
                                )[:, 0]
                                boundaries_y[(key[0], key[1])] = self.encoder_fn(
                                    boundary, key[0]
                                )[:, 1]
                                boundaries_x[(key[1], key[0])] = self.encoder_fn(
                                    boundary, key[1]
                                )[:, 0]
                                boundaries_y[(key[1], key[0])] = self.encoder_fn(
                                    boundary, key[1]
                                )[:, 1]

                        fig = plot_domains(
                            x=x,
                            y=y,
                            boundaries_x=boundaries_x,
                            boundaries_y=boundaries_y,
                            name=self.figure_path + f"/charts_{step}.png",
                        )
                        wandb.log({"charts": fig})

                new_progress = int(step * 100 / num_steps)
                if new_progress > progress:
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss})
                    progress = new_progress
                if step % self.save_every == 0:
                    self.save_model(step=step + 1)
                if step >= num_steps:
                    break
                step += 1
        self.save_model(step=step)

    def save_model(self, step=0):
        self.checkpointer.save_checkpoint(step=step, params=self.state.params)

    def load_model(self, step=None):
        if step is None:
            step = self.checkpointer.get_latest_checkpoint()
        self.state = self.state.replace(
            params=self.checkpointer.load_checkpoint(step=step)
        )

    def encoder_fn(self, x, idx) -> jnp.ndarray:
        """
        Returns the latent representation of the input batch

        Args:

            chart (jnp.ndarray): The input batch
            idx (int): The index of the chart

        Returns:
            jnp.ndarray: The latent representation of the input batch

        """
        params = jax.tree_map(lambda x: x[idx], self.state.params)
        return self.model.apply({"params": params}, x)[1]

    def autoencoder_fn(self, x, idx) -> jnp.ndarray:
        """
        Returns the reconstructed batch

        Args:

            chart (jnp.ndarray): The input batch
            idx (int): The index of the chart

        Returns:
            jnp.ndarray: The reconstructed batch

        """
        params = jax.tree_map(lambda x: x[idx], self.state.params)
        return self.model.apply({"params": params}, x)[0]


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
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    for i in range(num_plots):
        # Calculate row and column index for the plot
        row, col = divmod(i, cols)

        ax[row][col].set_title(f"Chart {i}")
        ax[row][col].scatter(x[i], y[i], s=3, c="b")

        for key in boundaries_x.keys():
            if key[0] == i:
                ax[row][col].scatter(
                    boundaries_x[key], boundaries_y[key], s=10, label=f"boundary {key}"
                )

        ax[row][col].legend(loc="best")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_3d_points(x, y, z, title="3D Points"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="r", marker="o")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig
