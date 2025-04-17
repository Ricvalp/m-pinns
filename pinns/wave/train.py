from pathlib import Path

import ml_collections
import models
from tqdm import tqdm
from jax.tree_util import tree_map
from samplers import (
    UniformICSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from chart_autoencoder.riemann import get_metric_tensor_and_sqrt_det_g_autodecoder
from chart_autoencoder.utils import prefetch_to_device

from pinns.wave.get_dataset import get_dataset

from pinns.wave.plot import (
    plot_domains,
    plot_domains_3d,
    plot_domains_with_metric,
    plot_combined_3d_with_metric,
)

import wandb
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, load_config

from utils import set_profiler


def train_and_evaluate(config: ml_collections.ConfigDict):

    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)
    Path(config.profiler.log_dir).mkdir(parents=True, exist_ok=True)
    logger = Logger()

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), d_params = get_metric_tensor_and_sqrt_det_g_autodecoder(
        autoencoder_config,
        step=config.autoencoder_checkpoint.step,
        inverse=True,
    )

    x, y, u0, u0_derivative, boundaries_x, boundaries_y, charts3d = get_dataset(
        autoencoder_config.dataset.charts_path,
        sigma=config.sigma_ics,
        amplitude=config.amplitude_ics,
    )

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

        plot_domains_3d(
            x,
            y,
            ics=u0_derivative,
            decoder=decoder,
            d_params=d_params,
            name=Path(config.figure_path) / "domains_3d_derivative.png",
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
            load_existing_batches=config.training.load_existing_batches,
            ics_path=(
                config.training.ics_batches_path,
                config.training.ics_values_path,
            )
        )
    )

    ics_derivative_sampler = iter(
        UniformICSampler(
            x=x,
            y=y,
            u0=u0_derivative,
            batch_size=config.training.batch_size,
            load_existing_batches=config.training.load_existing_batches,
            ics_path=(
                config.training.ics_derivative_batches_path,
                config.training.ics_derivative_values_path,
            )
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            sigma=config.training.uniform_sampler_sigma,
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
            load_existing_batches=config.training.load_existing_batches,
            boundary_batches_paths=(
                config.training.boundary_batches_path,
                config.training.boundary_pairs_idxs_path,
            ),
        )
    )

    model = models.Wave(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        num_charts=len(u0),
    )

    print("Waiting for JIT...")

    for step in tqdm(range(1, config.training.max_steps + 1), desc="Training"):

        set_profiler(config.profiler, step, config.profiler.log_dir)

        batch = next(res_sampler), next(boundary_sampler), next(ics_sampler), next(ics_derivative_sampler)
        loss, model.state = model.step(model.state, batch)

        if step % config.wandb.log_every_steps == 0:
            wandb.log({"loss": loss}, step)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)
        
        if step % config.wandb.eval_every_steps == 0:
            losses, _ = model.eval(model.state, batch)
            wandb.log({
                "ics_loss": losses["ics"],
                "bc_loss": losses["bc"],
                "ics_derivative_loss": losses["ics_derivative"],
                "res_loss": losses["res"],
                "ics_weight": model.state.weights['ics'],
                "ics_derivative_weight": model.state.weights['ics_derivative'],
                "res_weight": model.state.weights['res'],
                }, step)

        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                save_checkpoint(
                    model.state,
                    config.saving.checkpoint_dir,
                    keep=config.saving.num_keep_ckpts,
                )

    # for step in tqdm(range(step, step + config.training.lbfgs_max_steps + 1), desc="L-BFGS"):

    #     # set_profiler(config.profiler, step, config.profiler.log_dir)

    #     batch = next(res_sampler), next(boundary_sampler), next(ics_sampler)
    #     loss, model.state = model.lbfgs_step(model.state, batch)

    #     if step % config.wandb.log_every_steps == 0:
    #         wandb.log({"loss": loss}, step)

    return model
