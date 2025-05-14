from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from matplotlib import pyplot as plt

from jaxpi.evaluator import BaseEvaluator
from jaxpi.models import MPINN


class Diffusion(MPINN):
    def __init__(
        self, config, inv_metric_tensor, sqrt_det_g, d_params, ics, boundaries
    ):
        super().__init__(config, num_charts=len(ics[-1]))

        self.sqrt_det_g = sqrt_det_g
        self.inv_metric_tensor = inv_metric_tensor
        self.d_params = d_params

        self.x, self.y, self.ic = ics
        self.boundaries_x, self.boundaries_y = boundaries

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, None))

    def create_losses(self):
        @partial(vmap, in_axes=(0, 0, 0, 0))
        def compute_ics_loss(params, x, y, ics):
            u_pred = vmap(self.u_net, (None, 0, 0, None))(params, x, y, 0.0)
            return jnp.mean((u_pred - ics) ** 2)

        @partial(vmap, in_axes=(0, 0, 0))
        def compute_res_loss(params, d_params, res_batches):
            x, y, t = res_batches[:, 0], res_batches[:, 1], res_batches[:, 2]
            r_pred = vmap(self.r_net, (None, None, 0, 0, 0))(params, d_params, x, y, t)
            return jnp.mean(r_pred**2)

        @partial(vmap, in_axes=(None, 0, 0))
        @partial(vmap, in_axes=(None, 0, 0))
        def compute_boundary_loss(params, boundary_batches, boundary_idxs):
            xa, ya, t = (
                boundary_batches[0, :, 0],
                boundary_batches[0, :, 1],
                boundary_batches[0, :, 2],
            )
            xb, yb, t = (
                boundary_batches[1, :, 0],
                boundary_batches[1, :, 1],
                boundary_batches[1, :, 2],
            )

            a = boundary_idxs[0]
            b = boundary_idxs[1]

            boundary_pred_a = vmap(self.u_net, (None, 0, 0, 0))(
                jax.tree_map(lambda x: x[a], params), xa, ya, t
            )
            boundary_pred_b = vmap(self.u_net, (None, 0, 0, 0))(
                jax.tree_map(lambda x: x[b], params), xb, yb, t
            )

            return jnp.mean(0.25 * (boundary_pred_a - boundary_pred_b) ** 2)

        self.compute_ics_loss = compute_ics_loss
        self.compute_res_loss = compute_res_loss
        self.compute_boundary_loss = compute_boundary_loss

    def u_net(self, params, x, y, t):
        z = jnp.stack([x, y, t])
        u = self.state.apply_fn(params, z)
        return u[0]

    def g_inv_net(self, d_params, x, y):
        p = jnp.stack([x, y])[None, :]
        return self.inv_metric_tensor(d_params, p)[0]

    def sqrt_det_g_net(self, d_params, x, y):
        p = jnp.stack([x, y])[None, :]
        return self.sqrt_det_g(d_params, p)[0]  # check this!

    def laplacian_net(self, params, d_params, x, y, t):
        F1 = lambda x, y, t: self.sqrt_det_g_net(d_params, x, y) * (
            self.g_inv_net(d_params, x, y)[0, 0]
            * grad(self.u_net, argnums=1)(params, x, y, t)
            + self.g_inv_net(d_params, x, y)[0, 1]
            * grad(self.u_net, argnums=2)(params, x, y, t)
        )
        F2 = lambda x, y, t: self.sqrt_det_g_net(d_params, x, y) * (
            self.g_inv_net(d_params, x, y)[1, 0]
            * grad(self.u_net, argnums=1)(params, x, y, t)
            + self.g_inv_net(d_params, x, y)[1, 1]
            * grad(self.u_net, argnums=2)(params, x, y, t)
        )
        F1_x = grad(F1, argnums=0)(x, y, t)
        F2_y = grad(F2, argnums=1)(x, y, t)
        return (1.0 / self.sqrt_det_g_net(d_params, x, y)) * (F1_x + F2_y)

    def r_net(self, params, d_params, x, y, t):
        u_t = grad(self.u_net, argnums=3)(params, x, y, t)
        return u_t - 0.4 * self.laplacian_net(params, d_params, x, y, t)

    def losses(self, params, batch):

        res_batches, boundary_batches, ics_batches = batch

        ics_input_points, ics_values = ics_batches
        x, y = ics_input_points[:, :, 0], ics_input_points[:, :, 1]

        ics_loss = self.compute_ics_loss(params, x, y, ics_values)
        ics_loss = jnp.mean(ics_loss)

        res_loss = self.compute_res_loss(params, self.d_params, res_batches)
        res_loss = jnp.mean(res_loss)

        boundary_loss = self.compute_boundary_loss(
            params, boundary_batches[0], boundary_batches[1]
        )
        boundary_loss = jnp.mean(boundary_loss)

        loss_dict = {"ics": ics_loss, "res": res_loss, "bc": boundary_loss}

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.x, self.y)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class DiffusionEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.x, self.model.y)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
