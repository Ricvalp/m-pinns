from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, vmap
from matplotlib import pyplot as plt

from jaxpi.evaluator import BaseEvaluator
from jaxpi.models import MPINN


class Eikonal(MPINN):
    def __init__(
        self,
        config,
        inv_metric_tensor,
        sqrt_det_g,
        d_params,
        bcs_charts,
        boundaries,
        num_charts,
    ):
        super().__init__(config, num_charts=num_charts)

        self.sqrt_det_g = sqrt_det_g
        self.inv_metric_tensor = inv_metric_tensor
        self.d_params = d_params

        self.bcs_charts = bcs_charts
        self.boundaries_x, self.boundaries_y = boundaries

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))

    def create_losses(self):
        @partial(vmap, in_axes=(0, 0, 0, 0))
        def compute_bcs_loss(params, x, y, bcs):
            u_pred = vmap(self.u_net, (None, 0, 0))(params, x, y)
            return jnp.mean((u_pred - bcs) ** 2)

        @partial(vmap, in_axes=(0, 0, 0))
        def compute_res_loss(params, d_params, res_batches):
            x, y = res_batches[:, 0], res_batches[:, 1]
            r_pred = vmap(self.r_net, (None, None, 0, 0))(params, d_params, x, y)
            return jnp.mean(r_pred**2)

        @partial(vmap, in_axes=(None, 0, 0))
        @partial(vmap, in_axes=(None, 0, 0))
        def compute_boundary_loss(params, boundary_batches, boundary_idxs):
            xa, ya = (
                boundary_batches[0, :, 0],
                boundary_batches[0, :, 1],
            )
            xb, yb = (
                boundary_batches[1, :, 0],
                boundary_batches[1, :, 1],
            )

            a = boundary_idxs[0]
            b = boundary_idxs[1]

            boundary_pred_a = vmap(self.u_net, (None, 0, 0))(
                jax.tree_map(lambda x: x[a], params),
                xa,
                ya,
            )
            boundary_pred_b = vmap(self.u_net, (None, 0, 0))(
                jax.tree_map(lambda x: x[b], params),
                xb,
                yb,
            )

            return jnp.mean(0.25 * (boundary_pred_a - boundary_pred_b) ** 2)

        self.compute_bcs_loss = compute_bcs_loss
        self.compute_res_loss = compute_res_loss
        self.compute_boundary_loss = compute_boundary_loss

    def u_net(self, params, x, y):
        z = jnp.stack([x, y])
        u = self.state.apply_fn(params, z)
        return u[0]

    def g_inv_net(self, d_params, x, y):
        p = jnp.stack([x, y])[None, :]
        return self.inv_metric_tensor(d_params, p)[0]

    def sqrt_det_g_net(self, d_params, x, y):
        p = jnp.stack([x, y])[None, :]
        return self.sqrt_det_g(d_params, p)[0]  # check this!

    def square_norm_grad_u_net(self, params, d_params, x, y):

        g_ij = self.g_inv_net(d_params, x, y)
        dx_u = grad(self.u_net, argnums=1)(params, x, y)
        dy_u = grad(self.u_net, argnums=2)(params, x, y)

        norm_grad_u = (
            g_ij[0, 0] * dx_u**2 + g_ij[1, 1] * dy_u**2 + 2 * g_ij[0, 1] * dx_u * dy_u
        )
        return norm_grad_u

    def r_net(self, params, d_params, x, y):

        return self.square_norm_grad_u_net(params, d_params, x, y) - 1

    def losses(self, params, batch):

        res_batches, boundary_batches, bcs_batches = batch

        bcs_input_points, bcs_values = bcs_batches
        x, y = bcs_input_points[:, :, 0], bcs_input_points[:, :, 1]

        bcs_loss = self.compute_bcs_loss(params, x, y, bcs_values)
        bcs_loss = bcs_loss[self.bcs_charts]
        bcs_loss = jnp.mean(bcs_loss)

        res_loss = self.compute_res_loss(params, self.d_params, res_batches)
        res_loss = jnp.mean(res_loss)

        boundary_loss = self.compute_boundary_loss(
            params, boundary_batches[0], boundary_batches[1]
        )
        boundary_loss = jnp.mean(boundary_loss)

        loss_dict = {"bcs": bcs_loss, "res": res_loss, "bc": boundary_loss}

        return loss_dict

    def compute_l2_error(self, params, x, y, u_test, bcs_charts):
        error = self.compute_bcs_loss(params, x, y, u_test)
        error = error[bcs_charts]
        error = jnp.mean(error)
        return error
    

class EikonalEvaluator(BaseEvaluator):
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
