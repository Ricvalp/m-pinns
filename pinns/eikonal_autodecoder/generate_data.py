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
        idxs=config.idxs,
    )

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)

    Path(config.training.batches_path).mkdir(parents=True, exist_ok=True)

    plot_charts_solution(
        bcs_x,
        bcs_y,
        bcs,
        name=config.figure_path + "/generated_eikonal_train_bcs.png",
        vmin=0.0,
        vmax=1.5,
    )

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

    for step in tqdm(range(1, 501), desc="Generating batches"):

        # batch = next(res_sampler), next(boundary_sampler), next(bcs_sampler)
        # res_batches.append(batch[0])

        batch = None, next(boundary_sampler)  # next(bcs_sampler)

        boundary_batches.append(batch[1][0])
        boundary_pairs_idxs.append(batch[1][1])
        # bcs_batches.append(batch[2][0])
        # bcs_values.append(batch[2][1])

    # res_batches_array = np.array(res_batches)
    boundary_batches_array = np.array(boundary_batches)
    boundary_pairs_idxs_array = np.array(boundary_pairs_idxs)
    # bcs_batches_array = np.array(bcs_batches)
    # bcs_values_array = np.array(bcs_values)

    # np.save(config.training.batches_path + "res_batches.npy", res_batches_array)
    np.save(
        config.training.batches_path + "boundary_batches.npy", boundary_batches_array
    )
    np.save(
        config.training.batches_path + "boundary_pairs_idxs.npy",
        boundary_pairs_idxs_array,
    )

    # np.save(config.training.batches_path + "bcs_batches.npy", bcs_batches_array)
    # np.save(config.training.values_path + "bcs_values.npy", bcs_values_array)

    # print("Size of res_batches in MB: ", res_batches_array.nbytes / 1024 / 1024)
    print(
        "Size of boundary_batches in MB: ",
        boundary_batches_array.nbytes / 1024 / 1024,
    )
    print(
        "Size of boundary_pairs_idxs in MB: ",
        boundary_pairs_idxs_array.nbytes / 1024 / 1024,
    )
    # print("Size of bcs_batches in MB: ", bcs_batches_array.nbytes / 1024 / 1024)
    # print("Size of bcs_values in MB: ", bcs_values_array.nbytes / 1024 / 1024)

    # if step % 100 == 0:
    #     res_batches_array = np.array(res_batches)
    #     boundary_batches_array = np.array(boundary_batches)
    #     boundary_pairs_idxs_array = np.array(boundary_pairs_idxs)
    #     bcs_batches_array = np.array(bcs_batches)
    #     bcs_values_array = np.array(bcs_values)

    #     np.save(config.training.res_batches_path, res_batches_array)
    #     np.save(config.training.boundary_batches_path, boundary_batches_array)
    #     np.save(config.training.boundary_pairs_idxs_path, boundary_pairs_idxs_array)
    #     np.save(config.training.bcs_batches_path, bcs_batches_array)
    #     np.save(config.training.bcs_values_path, bcs_values_array)

    #     print("Size of res_batches in MB: ", res_batches_array.nbytes / 1024 / 1024)
    #     print(
    #         "Size of boundary_batches in MB: ",
    #         boundary_batches_array.nbytes / 1024 / 1024,
    #     )
    #     print(
    #         "Size of boundary_pairs_idxs in MB: ",
    #         boundary_pairs_idxs_array.nbytes / 1024 / 1024,
    #     )
    #     print("Size of bcs_batches in MB: ", bcs_batches_array.nbytes / 1024 / 1024)
    #     print("Size of bcs_values in MB: ", bcs_values_array.nbytes / 1024 / 1024)
