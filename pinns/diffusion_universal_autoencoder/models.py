from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from matplotlib import pyplot as plt

from jaxpi.evaluator import BaseEvaluator
from jaxpi.models import MPINNSingleChart


class DiffusionTime(MPINNSingleChart):
    def __init__(
        self, config, inv_metric_tensor, sqrt_det_g, conditionings, ts, ics 
    ):
        super().__init__(config)

        self.sqrt_det_g = sqrt_det_g
        self.inv_metric_tensor = inv_metric_tensor
        self.conditionings = conditionings
        self.ts = jnp.array(ts)

        self.x, self.y, self.ic = ics

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, None))

    def create_losses(self):
        # @partial(vmap, in_axes=(0, 0, 0, 0))
        def compute_ics_loss(params, x, y, ics):
            u_pred = vmap(self.u_net, (None, 0, 0, None))(params, x, y, 0.0)
            return jnp.mean((u_pred - ics) ** 2)

        # @partial(vmap, in_axes=(0, 0))
        def compute_res_loss(params, res_batches):
            res_batches, ts_idxs = res_batches
            x, y, t = res_batches[:, 0], res_batches[:, 1], self.ts[ts_idxs]
            conditioning = self.conditionings[ts_idxs]
            r_pred = vmap(self.r_net, (None, 0, 0, 0, 0))(params, conditioning, x, y, t)
            return jnp.mean(r_pred**2)

        self.compute_ics_loss = compute_ics_loss
        self.compute_res_loss = compute_res_loss

    def u_net(self, params, x, y, t):
        z = jnp.stack([x, y, t])
        u = self.state.apply_fn(params, z)
        return u[0]

    def g_inv_net(self, conditioning, x, y):
        p = jnp.stack([x, y])[None, :]
        return self.inv_metric_tensor(conditioning, p)[0]

    def sqrt_det_g_net(self, conditioning, x, y):
        p = jnp.stack([x, y])[None, :]
        return self.sqrt_det_g(conditioning, p)[0]

    def laplacian_net(self, params, conditioning, x, y, t):
        F1 = lambda x, y, t: self.sqrt_det_g_net(conditioning, x, y) * (
            self.g_inv_net(conditioning, x, y)[0, 0]
            * grad(self.u_net, argnums=1)(params, x, y, t)
            + self.g_inv_net(conditioning, x, y)[0, 1]
            * grad(self.u_net, argnums=2)(params, x, y, t)
        )
        F2 = lambda x, y, t: self.sqrt_det_g_net(conditioning, x, y) * (
            self.g_inv_net(conditioning, x, y)[1, 0]
            * grad(self.u_net, argnums=1)(params, x, y, t)
            + self.g_inv_net(conditioning, x, y)[1, 1]
            * grad(self.u_net, argnums=2)(params, x, y, t)
        )
        F1_x = grad(F1, argnums=0)(x, y, t)
        F2_y = grad(F2, argnums=1)(x, y, t)
        return (1.0 / self.sqrt_det_g_net(conditioning, x, y)) * (F1_x + F2_y)

    def r_net(self, params, conditioning, x, y, t):
        u_t = grad(self.u_net, argnums=3)(params, x, y, t)
        return u_t - 0.1 * self.laplacian_net(params, conditioning, x, y, t)

    def losses(self, params, batch):

        res_batches, ics_batches = batch

        ics_input_points, ics_values = ics_batches
        x, y = ics_input_points[:, 0], ics_input_points[:, 1]

        ics_loss = self.compute_ics_loss(params, x, y, ics_values)
        ics_loss = jnp.mean(ics_loss)

        res_loss = self.compute_res_loss(params, res_batches)
        res_loss = jnp.mean(res_loss)

        loss_dict = {"ics": ics_loss, "res": res_loss}

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
