from pathlib import Path

import ml_collections
from tqdm import tqdm
from samplers import (
    UniformICSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from jaxpi.utils import load_config

from pinns.wave.get_dataset import get_dataset

import numpy as np


def generate_data(config: ml_collections.ConfigDict):

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    x, y, u0, u0_derivative, boundaries_x, boundaries_y, charts3d = get_dataset(
        autoencoder_config.dataset.charts_path,
        sigma=config.sigma_ics,
    )

    ics_sampler = iter(
        UniformICSampler(
            x=x,
            y=y,
            u0=u0,
            batch_size=config.training.batch_size,
            load_existing_batches=False,
            ics_path=config.training.ics_batches_path,
        )
    )

    ics_derivative_sampler = iter(
        UniformICSampler(
            x=x,
            y=y,
            u0=u0_derivative,
            batch_size=config.training.batch_size,
            load_existing_batches=False,
            ics_path=config.training.ics_derivative_batches_path,

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
            load_existing_batches=False,
            boundary_batches_paths=(
                config.training.boundary_batches_path,
                config.training.boundary_pairs_idxs_path,
            ),
        )
    )

    res_batches = []
    boundary_batches = []
    boundary_pairs_idxs = []
    ics_batches = []
    ics_values = []
    ics_derivative_batches = []
    ics_derivative_values = []

    for step in tqdm(range(1, 100001), desc="Generating batches"):

        batch = next(res_sampler), next(boundary_sampler), next(ics_sampler), next(ics_derivative_sampler)
        res_batches.append(batch[0])
        boundary_batches.append(batch[1][0])
        boundary_pairs_idxs.append(batch[1][1])
        ics_batches.append(batch[2][0])
        ics_values.append(batch[2][1])
        ics_derivative_batches.append(batch[3][0])
        ics_derivative_values.append(batch[3][1])
        if step % 1000 == 0:
            res_batches_arrey = np.array(res_batches)
            boundary_batches_arrey = np.array(boundary_batches)
            boundary_pairs_idxs_arrey = np.array(boundary_pairs_idxs)
            ics_batches_arrey = np.array(ics_batches)
            ics_values_arrey = np.array(ics_values)
            ics_derivative_batches_arrey = np.array(ics_derivative_batches)
            ics_derivative_values_arrey = np.array(ics_derivative_values)
            np.save(config.training.res_batches_path, res_batches_arrey)
            np.save(config.training.boundary_batches_path, boundary_batches_arrey)
            np.save(config.training.boundary_pairs_idxs_path, boundary_pairs_idxs_arrey)
            np.save(config.training.ics_batches_path, ics_batches_arrey)
            np.save(config.training.ics_derivative_batches_path, ics_derivative_batches_arrey)
            np.save(config.training.ics_values_path, ics_values_arrey)
            np.save(config.training.ics_derivative_values_path, ics_derivative_values_arrey)
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
            print("Size of ics_derivative_batches in MB: ", ics_derivative_batches_arrey.nbytes / 1024 / 1024)
