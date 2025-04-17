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
from chart_autoencoder.models import AutoDecoder, Decoder
from chart_autoencoder.utils import ModelCheckpoint, set_profiler


class TrainerAutoDecoder:

    def __init__(
        self,
        cfg,
        chart,
        boundary_indices,
        chart_3d,
        distances_matrix,
        chart_key,
    ):
        super().__init__()

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
        self.num_steps = cfg.train.num_steps
        self.chart_3d = chart_3d
        self.chart_key = chart_key
        self.distances_matrix = distances_matrix
        self.lambda_reg_decay = cfg.train.reg_lambda_decay
        self.noise_scale_riemannian = cfg.train.noise_scale_riemannian
        self.lambda_geo_loss = cfg.train.lambda_geo_loss

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
        self.chart = chart
        self.boundary_indices = boundary_indices

        self.rng = jax.random.PRNGKey(cfg.seed)

        self.init_model()
        self.create_functions()

    def init_model(self):
        self.states = []
        self.rng, rng = jax.random.split(self.rng)
        model = AutoDecoder(
            init_points=self.chart,
            n_hidden=self.cfg_model.n_hidden,
            rff_dim=self.cfg_model.rff_dim,
        )

        decoder = Decoder(
            n_hidden=self.cfg_model.n_hidden,
            rff_dim=self.cfg_model.rff_dim,
            n_out=3,
        )
        self.decoder_apply_fn = decoder.apply

        # optimizer = optax.adamw(self.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.001)
        optimizer = optax.adam(self.lr, b1=0.9, b2=0.999, eps=1e-8)
        params = model.init(
            rng,
        )["params"]

        warmup_decoder_params = decoder.init(rng, params["points"])["params"]
        warmup_decoder_optimizer = optax.adamw(
            self.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.001
        )

        self.warmup_state = train_state.TrainState.create(
            apply_fn=decoder.apply,
            params=warmup_decoder_params,
            tx=warmup_decoder_optimizer,
        )

        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )

    def create_functions(self):

        if self.reg == "reg":

            def loss_fn(params, batch, key):
                out = self.state.apply_fn({"params": params})
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                riemannian_loss = riemannian_metric_loss(params, key)
                return jnp.mean(loss) + self.reg_lambda * riemannian_loss, (
                    riemannian_loss,
                    loss,
                )

        # elif self.reg == "reg+iso":

        #     def loss_fn(params, batch):
        #         batch, _ = batch
        #         out, z = self.model.apply({"params": params}, batch)
        #         loss = jnp.sum((out - batch) ** 2, axis=-1)
        #         reg = jnp.sum((z - self.center) ** 2, axis=-1)
        #         iso_loss = isometry_loss(batch, z)
        #         return jnp.mean(loss) + self.reg_lambda * (jnp.mean(reg) + iso_loss)

        elif self.reg == "reg+geo":

            def loss_fn(params, batch, reg_lambda, key):
                out = self.state.apply_fn({"params": params})
                loss = jnp.sum((out - batch) ** 2, axis=-1).mean()
                riemannian_loss = riemannian_metric_loss(params, key)
                geo_loss = geodesic_preservation_loss(
                    self.distances_matrix, params["points"]
                )
                return loss + reg_lambda * (riemannian_loss + self.lambda_geo_loss * geo_loss), (
                    riemannian_loss,
                    geo_loss,
                    loss,
                )

        # elif self.reg == 'reg+iso+geo':

        #     def loss_fn(params, batch):
        #         batch, distances_matrix = batch
        #         out, z = self.model.apply({"params": params}, batch)
        #         loss = jnp.sum((out - batch) ** 2, axis=-1)
        #         reg = jnp.sum((z - self.center) ** 2, axis=-1)
        #         iso_loss = isometry_loss(batch, z)
        #         geo_loss = geodesic_preservation_loss(distances_matrix, z)
        #         return jnp.mean(loss) + self.reg_lambda * (
        #             jnp.mean(reg) + iso_loss + 10*geo_loss
        #         )

        elif self.reg == "none":

            def loss_fn(params, batch):
                out = self.state.apply_fn({"params": params})
                loss = jnp.sum((out - batch) ** 2, axis=-1)
                return jnp.mean(loss)

        else:
            raise ValueError(f"Regularization method {self.reg} not defined")

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
            geodesic_dist = distances_matrix / jnp.mean(distances_matrix)
            return jnp.mean((z_dist - geodesic_dist) ** 2)

        def isometry_loss(batch, z):
            diff = batch[:, None, :] - batch[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
            z_diff = z[:, None, :] - z[None, :, :]
            z_dist = jnp.sqrt(jnp.sum(z_diff**2, axis=-1) + 1e-8)
            return jnp.mean((dist - z_dist) ** 2)

        def riemannian_metric_loss(params, key):
            d = lambda x: self.decoder_apply_fn({"params": params["D"]}, x)
            noise = (
                jax.random.normal(key, shape=params["points"].shape)
                * self.noise_scale_riemannian
            )
            points = params["points"] + noise
            J = vmap(jax.jacfwd(d))(points)
            J_T = jnp.transpose(J, (0, 2, 1))
            g = jnp.matmul(J_T, J)
            g_inv = jnp.linalg.inv(g)
            return jnp.mean(jnp.absolute(g)) + 0.1 * jnp.mean(jnp.absolute(g_inv))

        def train_step(state, batch, reg_lambda, key):
            my_loss = lambda params: loss_fn(params, batch, reg_lambda, key)
            (loss, aux), grads = jax.value_and_grad(my_loss, has_aux=True)(state.params)

            dgrads = jax.tree_util.tree_map(lambda x: x * 1.0, grads["D"])
            pgrads = grads["points"]
            grads = {"D": dgrads, "points": pgrads}
            state = state.apply_gradients(grads=grads)
            return state, loss, aux, grads

        def train_step_decoder_only(state, points, batch, key):
            my_loss = lambda params: loss_fn(
                {"D": params, "points": points}, batch, self.reg_lambda, key
            )

            (loss, aux), grads = jax.value_and_grad(my_loss, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss, aux

        self.train_step = jax.jit(train_step)  # train_step #
        self.train_step_decoder_only = jax.jit(train_step_decoder_only)

    def warmup(self, num_steps=1000):
        pbar = tqdm(range(num_steps))
        for step in pbar:
            self.rng, rng = jax.random.split(self.rng)
            self.warmup_state, loss, aux = self.train_step_decoder_only(
                self.warmup_state, self.state.params["points"], self.chart_3d, rng
            )
            pbar.set_description(f"Warmup loss: {loss:.4f}")
        self.state = self.state.replace(
            params={
                "D": self.warmup_state.params,
                "points": self.state.params["points"],
            }
        )

    def fit(self):

        pbar = tqdm(range(self.num_steps))
        reg_lambda = self.reg_lambda
        for step in pbar:
            # set_profiler(self.profiler_cfg, step, self.profiler_cfg.log_dir)
            self.rng, rng = jax.random.split(self.rng)
            reg_lambda = reg_lambda * self.lambda_reg_decay
            self.state, loss, aux, grads = self.train_step(
                self.state, self.chart_3d, reg_lambda, rng
            )
            pbar.set_description(f"Loss: {loss:.4f}")
            if step % self.log_charts_every == 0:
                x = [self.state.params["points"][:, 0]]
                y = [self.state.params["points"][:, 1]]
                boundaries_x = {}
                boundaries_y = {}
                for key in self.boundary_indices.keys():
                    boundary_indices = self.boundary_indices[key]
                    boundaries_x[(key[0], key[1])] = x[0][jnp.array(boundary_indices)]
                    boundaries_y[(key[0], key[1])] = y[0][jnp.array(boundary_indices)]
                fig = plot_domains(
                    x=x,
                    y=y,
                    boundaries_x=boundaries_x,
                    boundaries_y=boundaries_y,
                    name=self.figure_path / f"{step}.png",
                )
            if self.wandb_log and step % self.wandb_log_every == 0:
                wandb.log(
                    {
                        "loss": loss,
                        "points grads norm": jnp.linalg.norm(grads["points"]),
                        "decoder grads norm": jnp.linalg.norm(
                            jnp.concatenate(
                                [
                                    jnp.ravel(p)
                                    for p in jax.tree_util.tree_leaves(grads["D"])
                                ]
                            )
                        ),
                        "riemannian_loss": aux[0],
                        "geodesic_loss": aux[1],
                        "recon loss": aux[2],
                        "reg_lambda": reg_lambda,
                        "chart_key": self.chart_key,
                    }, step=step
                )
            if (step + 1) % self.save_every == 0:
                self.save_model(step=step - 1)
        self.save_model(step=step)

    def save_model(self, step):
        self.checkpointer.save_checkpoint(step=step, params=self.state.params)

    def load_model(self, step=None):
        if step is None:
            step = self.checkpointer.get_latest_checkpoint()
        self.state = self.state.replace(
            params=self.checkpointer.load_checkpoint(step=step)
        )

    def decoder_fn(self) -> jnp.ndarray:
        return self.state.apply_fn({"params": self.state.params})


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
