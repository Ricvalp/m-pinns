from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

import jax
from flax.training import train_state
from flax import jax_utils

import jax.numpy as jnp
from jax import lax, jit, grad, random, jacfwd, jacrev, value_and_grad

from jax.tree_util import tree_map, tree_reduce, tree_leaves

from flax import struct
import optax

from jaxpi import archs
from jaxpi.utils import flatten_pytree


class TrainState(train_state.TrainState):
    weights: Dict
    momentum: float
    lbfgs_opt_state: Any
    lbfgs_lr: float
    lbfgs_tx: Any = struct.field(pytree_node=False)

    def apply_lbfgs_gradients(self, *, grads, **kwargs):
        """Updates


        Args:
        grads: The gradients to apply.

        Returns:
            An updated instance of `self` with new params and opt_state.

        """

        grads_with_opt = grads["params"]
        params_with_opt = self.params["params"]

        updates, new_opt_state = self.lbfgs_tx.update(
            grads_with_opt, self.lbfgs_opt_state, params_with_opt
        )
        # params_with_opt = params_with_opt - self.lbfgs_lr * updates
        params_with_opt = jax.tree.map(
            lambda x, y: x - self.lbfgs_lr * y,
            params_with_opt,
            updates,
        )
        new_params = params_with_opt

        return self.replace(
            step=self.step + 1,
            params={"params": new_params},
            lbfgs_opt_state=new_opt_state,
            **kwargs,
        )

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)

    elif config.arch_name == "ModifiedMlp":
        arch = archs.ModifiedMlp(**config)

    elif config.arch_name == "DeepONet":
        arch = archs.DeepONet(**config)

    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def _create_optimizer(config):
    if config.optimizer == "Adam":
        lr = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.decay_steps,
            decay_rate=config.decay_rate,
        )
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )
    elif config.optimizer == "AdamWarmupCosineDecay":
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            end_value=config.learning_rate / 1000,
        )
        tx = optax.adam(lr_schedule)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")

    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)

    return tx


def _create_train_state_multi_chart(config, num_charts):

    arch = _create_arch(config.arch)
    vmapped_init = jax.vmap(arch.init, in_axes=(0, None))
    x = jnp.ones(config.input_dim)
    params = vmapped_init(jax.random.split(random.PRNGKey(config.seed), num_charts), x)[
        "params"
    ]

    tx = _create_optimizer(config.optim)
    lbfgs_tx = optax.scale_by_lbfgs()

    init_weights = dict(config.weighting.init_weights)

    state = TrainState.create(
        apply_fn=arch.apply,
        params={"params": params},
        tx=tx,
        weights=init_weights,
        momentum=config.weighting.momentum,
        lbfgs_tx=lbfgs_tx,
        lbfgs_opt_state=lbfgs_tx.init(params),
        lbfgs_lr=config.optim.lbfgs_learning_rate,
    )

    return state


class MPINN:
    def __init__(self, config, num_charts):
        self.config = config
        self.state = _create_train_state_multi_chart(config, num_charts)
        self.create_functions()
        self.create_losses()

    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_l2_error(self, params, eval_x, eval_y, u_eval):
        raise NotImplementedError("Subclasses should implement this!")

    def loss(self, params, weights, batch, *args):
        # Compute losses
        losses = self.losses(params, batch, *args)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    def compute_weights(self, params, batch, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Compute the gradient of each loss w.r.t. the parameters
            grads = jacrev(self.losses)(params, batch, *args)

            # Compute the grad norm of each loss
            grad_norm_dict = {}
            for key, value in grads.items():
                flattened_grad = flatten_pytree(value)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            # Compute the mean of grad norms over all losses
            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
            # Grad Norm Weighting
            w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)

        # elif self.config.weighting.scheme == "ntk":
        #     # Compute the diagonal of the NTK of each loss
        #     ntk = self.compute_diag_ntk(params, batch, *args)

        #     # Compute the mean of the diagonal NTK corresponding to each loss
        #     mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

        #     # Compute the average over all ntk means
        #     mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
        #     # NTK Weighting
        #     w = tree_map(lambda x: (mean_ntk / x), mean_ntk_dict)

        return w

    def create_functions(self):

        def update_weights(state, batch, *args):
            weights = self.compute_weights(state.params, batch, *args)
            state = state.apply_weights(weights=weights)
            return state

        def step(state, batch, *args):
            loss, grads = value_and_grad(self.loss)(
                state.params, state.weights, batch, *args
            )
            state = state.apply_gradients(grads=grads)
            return loss, state

        def lbfgs_step(state, batch, *args):
            loss, grads = value_and_grad(self.loss)(
                state.params, state.weights, batch, *args
            )
            state = state.apply_lbfgs_gradients(grads=grads)
            return loss, state

        def eval(state, batch, eval_x=None, eval_y=None, u_eval=None, bcs_charts=None):
            losses = self.losses(state.params, batch)
            eval_loss = self.compute_l2_error(
                state.params, eval_x, eval_y, u_eval, bcs_charts
            )

            return losses, eval_loss

        self.update_weights = jax.jit(update_weights)
        self.step = jax.jit(step)
        self.lbfgs_step = jax.jit(lbfgs_step)
        self.eval = jax.jit(eval)

    def create_losses(self):
        raise NotImplementedError("Subclasses should implement this!")
