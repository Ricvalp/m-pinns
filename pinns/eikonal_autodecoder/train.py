from pathlib import Path
import matplotlib.pyplot as plt

import ml_collections
import models
from tqdm import tqdm
from samplers import (
    UniformBCSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from jax import config
config.update("jax_enable_x64", True)

import pandas as pd
import os

import json
import logging

import jax
import jax.numpy as jnp

from chart_autoencoder import (
    get_metric_tensor_and_sqrt_det_g_autodecoder,
    find_intersection_indices,
    find_closest_points_to_mesh,
    load_charts,
)

from pinns.eikonal_autodecoder.get_dataset import (
    get_dataset,
    get_eikonal_gt_solution,
)

from pinns.eikonal_autodecoder.plot import (
    plot_domains,
    plot_domains_3d,
    plot_domains_with_metric,
    plot_combined_3d_with_metric,
    plot_correlation,
)

import numpy as np

import wandb
from jaxpi.utils import save_checkpoint, load_config
from jaxpi.solution import get_final_solution

from utils import set_profiler


def train_and_evaluate(config: ml_collections.ConfigDict):

    wandb_config = config.wandb
    run = wandb.init(
        project=wandb_config.project,
        name=wandb_config.name,
        entity=wandb_config.entity,
        config=config,
    )

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)

    # Path(config.profiler.log_dir).mkdir(parents=True, exist_ok=True)

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    checkpoint_dir = f"{config.saving.checkpoint_dir}/{run.id}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir + "/cfg.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), d_params = get_metric_tensor_and_sqrt_det_g_autodecoder(
        autoencoder_config,
        step=config.autoencoder_checkpoint.step,
        inverse=True,
    )

    x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, charts3d = get_dataset(
        charts_path=autoencoder_config.dataset.charts_path,
        N=config.N,
        idxs=config.idxs,
    )

    np.save(
        autoencoder_config.dataset.charts_path + "/known_solution.npy",
        bcs,
        allow_pickle=True,
    )
    logging.info(f"Saved known solution")

    num_charts = len(x)

    if config.plot:

        # plot_domains(
        #     x,
        #     y,
        #     boundaries_x,
        #     boundaries_y,
        #     bcs_x=bcs_x,
        #     bcs_y=bcs_y,
        #     bcs=bcs,
        #     name=Path(config.figure_path) / "domains.png",
        # )

        # plot_domains_3d(
        #     x,
        #     y,
        #     bcs_x=bcs_x,
        #     bcs_y=bcs_y,
        #     bcs=bcs,
        #     decoder=decoder,
        #     d_params=d_params,
        #     name=Path(config.figure_path) / "domains_3d.png",
        # )

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

    bcs_sampler = iter(
        UniformBCSampler(
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            num_charts=len(x),
            batch_size=config.training.batch_size,
            # bcs_batches_path=(
            #     config.training.batches_path + "bcs_batches.npy",
            #     config.training.batches_path + "bcs_values.npy",
            # ),
            # load_existing_batches=config.training.load_existing_batches,
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            sigma=0.02,
            batch_size=config.training.batch_size,
        )
    )

    boundary_sampler = iter(
        UniformBoundarySampler(
            boundaries_x=boundaries_x,
            boundaries_y=boundaries_y,
            batch_size=config.training.batch_size,
            boundary_batches_paths=(
                config.training.batches_path + "boundary_batches.npy",
                config.training.batches_path + "boundary_pairs_idxs.npy",
            ),
            load_existing_batches=config.training.load_existing_batches,
        )
    )

    model = models.Eikonal(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        bcs_charts=jnp.array(list(bcs.keys())),
        boundaries=(boundaries_x, boundaries_y),
        num_charts=num_charts,
    )

    _, _, _, _, eval_x, eval_y, u_eval, _ = get_dataset(
        charts_path=autoencoder_config.dataset.charts_path,
        N=config.logging.num_eval_points,
    )
    max_eval_points = max([len(eval_x[key]) for key in eval_x.keys()])
    eval_idxs = {
        key: np.random.randint(0, len(eval_x[key]), max_eval_points)
        for key in eval_x.keys()
    }

    bcs_charts = jnp.array(list(u_eval.keys()))

    eval_x = jnp.array(
        [
            (
                jnp.array(eval_x[key][eval_idxs[key]])
                if key in eval_x.keys()
                else jnp.zeros((max_eval_points,))
            )
            for key in range(num_charts)
        ]
    )
    eval_y = jnp.array(
        [
            (
                jnp.array(eval_y[key][eval_idxs[key]])
                if key in eval_y.keys()
                else jnp.zeros((max_eval_points,))
            )
            for key in range(num_charts)
        ]
    )
    u_eval = jnp.array(
        [
            (
                jnp.array(u_eval[key][eval_idxs[key]])
                if key in u_eval.keys()
                else jnp.zeros((max_eval_points,))
            )
            for key in range(num_charts)
        ]
    )

    logging.info("Jitting...")

    for step in tqdm(range(1, config.training.max_steps + 1), desc="Training"):

        # set_profiler(config.profiler, step, config.profiler.log_dir)

        batch = next(res_sampler), next(boundary_sampler), next(bcs_sampler)
        loss, model.state = model.step(model.state, batch)

        if step % config.logging.log_every_steps == 0:
            wandb.log({"loss": loss}, step)

        if step % config.logging.eval_every_steps == 0:
            losses, eval_loss = model.eval(
                model.state, batch, eval_x, eval_y, u_eval, bcs_charts
            )
            wandb.log(
                {
                    "eval_loss": eval_loss,
                    "bcs_loss": losses["bcs"],
                    "res_loss": losses["res"],
                    "boundary_loss": losses["bc"],
                    "bcs_weight": model.state.weights["bcs"],
                    "res_weight": model.state.weights["res"],
                    "boundary_weight": model.state.weights["bc"],
                },
                step,
            )

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        if config.saving.save_every_steps is not None:
            if (
                step % config.saving.save_every_steps == 0
                or (step) == config.training.max_steps
            ):
                save_checkpoint(
                    model.state,
                    checkpoint_dir,
                    keep=config.saving.num_keep_ckpts,
                )

                mesh_sol, gt_sol = log_correlation(
                    config=config,
                    model=model,
                    x=x,
                    y=y,
                    params=model.state.params,
                    charts_path=autoencoder_config.dataset.charts_path,
                    step=step,
                )

    MSE = jnp.mean((mesh_sol - gt_sol) ** 2)
    corr = jnp.corrcoef(mesh_sol, gt_sol)[0, 1]

    logging.info(f"MSE: {MSE}")
    logging.info(f"Correlation: {corr}")

    write_to_csv(MSE, corr, config)

    return model


def log_correlation(
    config: ml_collections.ConfigDict,
    model: models.Eikonal,
    x: jnp.ndarray,
    y: jnp.ndarray,
    params: dict,
    step: int,
    charts_path: str,
):

    charts, charts_idxs, boundaries, boundary_indices, charts2d = load_charts(
        charts_path=charts_path,
    )

    u_preds = {}

    logging.info("Evaluating the solution on the charts")
    for i in tqdm(range(len(x))):
        u_preds[i] = model.u_pred_fn(jax.tree.map(lambda x: x[i], params), x[i], y[i])

    pts, sol = get_final_solution(
        charts=charts,
        charts_idxs=charts_idxs,
        u_preds=u_preds,
    )

    mesh_pts, gt_sol = get_eikonal_gt_solution(
        charts_path=charts_path,
    )

    # gt_sol_pts_idxs, _ = find_intersection_indices(
    #     mesh_pts,
    #     pts,
    # )

    _, gt_sol_pts_idxs = find_closest_points_to_mesh(
        mesh_pts,
        pts,
    )

    mesh_sol = sol[gt_sol_pts_idxs]

    fig = plot_correlation(
        mesh_sol, gt_sol, name=config.figure_path + f"/eikonal_correlation_{step}.png"
    )

    wandb.log({"correlation": fig}, step)

    plt.close(fig)

    return mesh_sol, gt_sol


def write_to_csv(MSE, corr, config: ml_collections.ConfigDict):

    csv_path = config.saving.csv_path
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Find the row that matches the current N and seed values
        N = config.N
        seed = config.seed
        mask = (df["N"] == N) & (df["seed"] == seed)

        if mask.any():
            # Update the existing row
            df.loc[mask, "mpinn_corr"] = float(corr)
            df.loc[mask, "mpinn_mse"] = float(MSE)
            df.to_csv(csv_path, index=False)
            logging.info(f"Updated CSV file for N={N}, seed={seed}")
        else:
            logging.warning(f"Row not found in CSV for N={N}, seed={seed}")
    else:
        logging.warning(f"CSV file {csv_path} not found")
