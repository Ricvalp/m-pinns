from pathlib import Path

import ml_collections
from tqdm import tqdm
from samplers import (
    UniformBCSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from jaxpi.utils import load_config

from pinns.eikonal_autodecoder.get_dataset import get_dataset
from pinns.eikonal_autodecoder.plot import plot_charts_solution
import numpy as np


def generate_data(config: ml_collections.ConfigDict):

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, charts3d = get_dataset(
        charts_path=autoencoder_config.dataset.charts_path,
        N=config.N,
    )
    
    Path(config.figure_path).mkdir(parents=True, exist_ok=True)
    plot_charts_solution(bcs_x, bcs_y, bcs, name=config.figure_path + "/generated_eikonal_train_bcs.png", vmin=0., vmax=1.5)

    bcs_sampler = iter(
        UniformBCSampler(
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            num_charts=len(x),
            batch_size=config.training.batch_size,
            load_existing_batches=False,
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
            load_existing_batches=False,
        )
    )

    res_batches = []
    boundary_batches = []
    boundary_pairs_idxs = []
    bcs_batches = []
    bcs_values = []

    for step in tqdm(range(1, 4001), desc="Generating batches"):

        batch = next(res_sampler), next(boundary_sampler), next(bcs_sampler)
        res_batches.append(batch[0])
        boundary_batches.append(batch[1][0])
        boundary_pairs_idxs.append(batch[1][1])
        bcs_batches.append(batch[2][0])
        bcs_values.append(batch[2][1])

        if step % 100 == 0:
            res_batches_arrey = np.array(res_batches)
            boundary_batches_arrey = np.array(boundary_batches)
            boundary_pairs_idxs_arrey = np.array(boundary_pairs_idxs)
            bcs_batches_arrey = np.array(bcs_batches)
            bcs_values_arrey = np.array(bcs_values)

            np.save(config.training.res_batches_path, res_batches_arrey)
            np.save(config.training.boundary_batches_path, boundary_batches_arrey)
            np.save(config.training.boundary_pairs_idxs_path, boundary_pairs_idxs_arrey)
            np.save(config.training.bcs_batches_path, bcs_batches_arrey)
            np.save(config.training.bcs_values_path, bcs_values_arrey)

            print("Size of res_batches in MB: ", res_batches_arrey.nbytes / 1024 / 1024)
            print(
                "Size of boundary_batches in MB: ",
                boundary_batches_arrey.nbytes / 1024 / 1024,
            )
            print(
                "Size of boundary_pairs_idxs in MB: ",
                boundary_pairs_idxs_arrey.nbytes / 1024 / 1024,
            )
            print("Size of bcs_batches in MB: ", bcs_batches_arrey.nbytes / 1024 / 1024)
            print("Size of bcs_values in MB: ", bcs_values_arrey.nbytes / 1024 / 1024)
