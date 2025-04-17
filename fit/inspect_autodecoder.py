from absl import app, logging
from ml_collections import ConfigDict
import json
import argparse
import tqdm
from pathlib import Path
import numpy as np
from datasets import get_dataset
from chart_autoencoder import (
    plot_3d_charts,
    plot_local_charts_2d,
    plot_html_3d_charts,
    get_charts,
)
import jax
from jax import jit, vmap
from chart_autoencoder.trainer_autodecoder import Decoder
from chart_autoencoder.utils import ModelCheckpoint
from chart_autoencoder.riemann import compute_norm_g_ginv_from_params

import argparse


parser = argparse.ArgumentParser()
# parser.add_argument("seed", type=int, help="Step number to load model from")
parser.add_argument(
    "dataset",
    type=str,
    help="Dataset to inspect",
    default="coil",
    nargs="?",
)
args = parser.parse_args()


def main(_):

    with open(f"./fit/checkpoints/{args.dataset}/cfg.json", "r") as f:
        cfg = ConfigDict(json.load(f))

    cfg = cfg.unlock()
    cfg.dataset.load_existing_charts = True
    cfg.dataset.use_existing_distance_matrix = True
    cfg.dataset.use_existing_umap_embeddings = True
    cfg.checkpoint.overwrite = False
    cfg.figure_path = f"./figures/inspect"

    train_data = get_dataset(cfg.dataset)

    charts, _, boundary_indices, _ = get_charts(
        points=train_data.data,
        cfg=cfg,
    )

    checkpoint_autodecoders = list(Path(cfg.checkpoint.checkpoint_path).iterdir())

    recon_charts = {}
    recon_noisy_charts = {}
    latent_charts = {}
    noisy_latent_charts = {}
    gs = {}
    g_invs = {}

    for checkpoint_autodecoder in tqdm(checkpoint_autodecoders):
        if checkpoint_autodecoder.name.startswith("chart"):
            checkpointer = ModelCheckpoint(
                path=Path(checkpoint_autodecoder).absolute(),
                max_to_keep=1,
                keep_every=1,
                overwrite=False,
            )

            key = checkpoint_autodecoder.name.split("_")[-1]
            params = checkpointer.load_checkpoint(cfg.train.num_steps - 1)

            decoder = Decoder(
                n_hidden=params["D"]["dense3"]["kernel"].shape[0],
                rff_dim=cfg.model.rff_dim,
                n_out=3,
            )
            (
                recon_noisy_chart,
                recon_chart,
                latent_chart,
                noisy_latent_chart,
                g,
                g_inv,
                norm_g,
                norm_g_inv,
            ) = compute_norm_g_ginv_from_params(
                params=params,
                decoder_fn=jax.jit(decoder.apply),
            )

            recon_noisy_charts[key] = recon_noisy_chart
            recon_charts[key] = recon_chart
            latent_charts[key] = params["points"]
            noisy_latent_charts[key] = noisy_latent_chart
            gs[key] = norm_g
            g_invs[key] = norm_g_inv

    plot_html_3d_charts(
        charts=recon_noisy_charts,
        original_charts=charts,
        g=gs,
        name=Path(cfg.figure_path)
        / f"post_{cfg.dataset.name}_noisy_charts_3d_with_g.html",
        sampled_points=None,
    )
    plot_html_3d_charts(
        charts=recon_noisy_charts,
        original_charts=charts,
        g=g_invs,
        name=Path(cfg.figure_path)
        / f"post_{cfg.dataset.name}_noisy_charts_3d_with_g_inv.html",
        sampled_points=None,
    )
    plot_3d_charts(
        charts=recon_charts,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_charts_3d.png",
    )

    plot_local_charts_2d(
        charts=noisy_latent_charts,
        original_charts=latent_charts,
        g=gs,
        boundaries_indices=boundary_indices,
        name=Path(cfg.figure_path)
        / f"post_{cfg.dataset.name}_noisy_latent_charts_with_g.png",
    )

    plot_local_charts_2d(
        charts=noisy_latent_charts,
        original_charts=latent_charts,
        g=g_invs,
        boundaries_indices=boundary_indices,
        name=Path(cfg.figure_path)
        / f"post_{cfg.dataset.name}_noisy_latent_charts_with_g_inv.png",
    )


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


if __name__ == "__main__":
    app.run(main)
