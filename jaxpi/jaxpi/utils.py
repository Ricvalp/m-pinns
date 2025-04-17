import os

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import jit, grad, tree_map
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from flax.training import checkpoints
from ml_collections import ConfigDict
import json


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


@partial(jit, static_argnums=(0,))
def jacobian_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = grad(apply_fn, argnums=0)(params, *args)
    J, _ = ravel_pytree(J)
    return J


@partial(jit, static_argnums=(0,))
def ntk_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = jacobian_fn(apply_fn, params, *args)
    K = jnp.dot(J, J)
    return K


def save_checkpoint(state, path, keep=5, name=None):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(path):
        os.makedirs(path)

    step = int(state.step)
    checkpoints.save_checkpoint(Path(path).absolute(), state, step=step, keep=keep)


def restore_checkpoint(state, path, step=None):
    state = checkpoints.restore_checkpoint(path, state, step=step)
    return state


def load_config(filename):
    with open(filename, "r") as f:
        cfg = json.load(f)
    return ConfigDict(cfg)
