from pathlib import Path

import ml_collections
from tqdm import tqdm
from samplers import (
    UniformICSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from jaxpi.utils import load_config

from pinns.diffusion_single_gpu_autodecoder.get_dataset import get_dataset

import numpy as np


def generate_data(config: ml_collections.ConfigDict):

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    x, y, u0, boundaries_x, boundaries_y, charts3d = get_dataset(
        autoencoder_config.dataset.charts_path,
        sigma=1.0,
    )

    ics_sampler = iter(
        UniformICSampler(
            x=x,
            y=y,
            u0=u0,
            batch_size=config.training.batch_size,
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
        )
    )

    res_batches = []
    boundary_batches = []
    boundary_pairs_idxs = []
    ics_batches = []
    ics_idxs = []

    for step in tqdm(range(1, 2001), desc="Generating batches"):

        batch = next(res_sampler), next(boundary_sampler), next(ics_sampler)
        res_batches.append(batch[0])
        boundary_batches.append(batch[1][0])
        boundary_pairs_idxs.append(batch[1][1])
        ics_batches.append(batch[2][0])
        ics_idxs.append(batch[2][1])

        if step % 100 == 0:
            res_batches_arrey = np.array(res_batches)
            boundary_batches_arrey = np.array(boundary_batches)
            boundary_pairs_idxs_arrey = np.array(boundary_pairs_idxs)
            ics_batches_arrey = np.array(ics_batches)
            ics_idxs_arrey = np.array(ics_idxs)

            np.save(config.training.res_batches_path, res_batches_arrey)
            np.save(config.training.boundary_batches_path, boundary_batches_arrey)
            np.save(config.training.boundary_pairs_idxs_path, boundary_pairs_idxs_arrey)
            np.save(config.training.ics_batches_path, ics_batches_arrey)
            np.save(config.training.ics_idxs_path, ics_idxs_arrey)

            print("Size of res_batches in MB: ", res_batches_arrey.nbytes / 1024 / 1024)
            print(
                "Size of boundary_batches in MB: ",
                boundary_batches_arrey.nbytes / 1024 / 1024,
            )
            print(
                "Size of boundary_pairs_idxs in MB: ",
                boundary_pairs_idxs_arrey.nbytes / 1024 / 1024,
            )
            print("Size of ics_batches in MB: ", ics_batches_arrey.nbytes / 1024 / 1024)
            print("Size of ics_idxs in MB: ", ics_idxs_arrey.nbytes / 1024 / 1024)
