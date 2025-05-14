import os
import time
from pathlib import Path

import json


import jax
import jax.numpy as jnp
import ml_collections
import models
from tqdm import tqdm
from jax.tree_util import tree_map
from samplers import (
    UniformICSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from chart_autoencoder import (
    get_metric_tensor_and_sqrt_det_g_grid_universal_autodecoder,
    load_charts3d,
)

from pinns.diffusion_universal_autoencoder.get_dataset import get_dataset

from pinns.diffusion_universal_autoencoder.plot import (
    plot_domains,
    plot_domains_3d,
    plot_domains_with_metric,
    plot_combined_3d_with_metric,
)

import wandb
from jaxpi.utils import save_checkpoint, load_config

from utils import set_profiler

import matplotlib.pyplot as plt
import numpy as np


def train_and_evaluate(config: ml_collections.ConfigDict):

    wandb_config = config.wandb
    run = wandb.init(
        project=wandb_config.project,
        name=wandb_config.name,
        entity=wandb_config.entity,
        config=config,
    )

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    checkpoint_dir = f"{config.saving.checkpoint_dir}/{run.id}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir + "/cfg.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    X = jnp.linspace(0, 1, 50)
    Y = jnp.linspace(0, 1, 50)
    XX, YY = jnp.meshgrid(X, Y)
    coords = jnp.zeros((XX.size, 3))
    coords[:, 0] = XX.flatten()
    coords[:, 1] = YY.flatten()

    boundaries_x = jnp.zeros((XX.size, 3))
    boundaries_x[:, 0] = XX.flatten()
    boundaries_x[:, 1] = YY.flatten()

    boundaries_y = jnp.zeros((XX.size, 3))
    boundaries_y[:, 0] = XX.flatten()
    boundaries_y[:, 1] = YY.flatten()
    
    charts3d = []
    ts = np.linspace(0.0, 0.8, 1000)
    for t in ts:
        charts3d.append(get_deformed_points(coords, t))

    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), (conditionings, d_params) = get_metric_tensor_and_sqrt_det_g_grid_universal_autodecoder(
        autoencoder_cfg=autoencoder_config,
        cfg=config,
        charts=charts3d,
        coords=coords,
        inverse=True,
    )

    x = coords[:, 0]
    y = coords[:, 1]

    def initial_conditions_spike(x, y, x0=0.5, y0=0.5, sigma=1.0, amplitude=10.0):
        return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)

    u0 = initial_conditions_spike(x, y)


    if config.plot:

        plot_domains(
            x,
            y,
            boundaries_x,
            boundaries_y,
            ics=u0,
            name=Path(config.figure_path) / "domains.png",
        )

        plot_domains_3d(
            x,
            y,
            ics=u0,
            decoder=decoder,
            d_params=d_params,
            name=Path(config.figure_path) / "domains_3d.png",
        )

        plot_domains_with_metric(
            x,
            y,
            sqrt_det_g,
            d_params=d_params,
            name=Path(config.figure_path) / "domains_with_metric.png",
        )

        plot_combined_3d_with_metric(
            x,
            y,
            decoder=decoder,
            sqrt_det_g=sqrt_det_g,
            d_params=d_params,
            name=Path(config.figure_path) / "combined_3d_with_metric.png",
        )


    ics_sampler = iter(
        UniformICSampler(
            x=x,
            y=y,
            u0=u0,
            batch_size=config.training.batch_size,
            ics_batches_path=(
                config.training.ics_batches_path,
                config.training.ics_idxs_path,
            ),
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            sigma=0.1,
            T=config.T,
            batch_size=config.training.batch_size,
        )
    )

    boundary_sampler = iter(
        UniformBoundarySampler(
            boundaries_x=boundaries_x,
            boundaries_y=boundaries_y,
            T=config.T,
            batch_size=config.training.batch_size,
            boundary_batches_paths=(
                config.training.boundary_batches_path,
                config.training.boundary_pairs_idxs_path,
            ),
        )
    )

    model = models.Diffusion(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        ics=(x, y, u0),
        boundaries=(boundaries_x, boundaries_y),
    )

    print("Waiting for JIT...")

    for step in tqdm(range(1, config.training.max_steps + 1), desc="Training"):

        # set_profiler(config.profiler, step, config.profiler.log_dir)

        batch = next(res_sampler), next(boundary_sampler), next(ics_sampler)
        loss, model.state = model.step(model.state, batch)

        if step % config.wandb.log_every_steps == 0:
            wandb.log({"loss": loss}, step)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                save_checkpoint(
                    model.state,
                    config.saving.checkpoint_dir,
                    keep=config.saving.num_keep_ckpts,
                )

    return model





def get_deformed_points(self, grid, t):
    """
    Apply a smooth deformation in the z-direction to the chart points.
    
    Args:
        chart_id: index of the chart to transform
        t: controls the strength of deformation (0.0 = no deformation)
    
    Returns:
        Transformed points
    """
    # Get the base points for this chart
    points = grid.copy()  # Make a copy to avoid modifying original data
    
    # Get x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Generate random frequencies (but still zero at boundaries)
    freq_x = np.array([1, 2])  # Random integer between 1 and 3
    freq_y = np.array([1, 2])
    
    # Create a 2D sine wave deformation in z-direction that is zero at the boundaries
    # Sum over various frequencies for a more complex deformation pattern
    deformation_z = np.zeros_like(x)
    for i in range(len(freq_x)):
        # Each term is zero when x=0, x=1, y=0, or y=1
        deformation_z += t[i] * np.sin(freq_x[i] * np.pi * x) * np.sin(freq_y[i] * np.pi * y)
    
    # Apply the deformation to z coordinate
    points[:, 2] = deformation_z
    
    return points