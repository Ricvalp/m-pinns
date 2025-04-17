import numpy as np
import orbax.checkpoint as ocp
import collections
import itertools
import logging
import networkx as nx
from torch.utils.data import Dataset
import jax
import flax
from typing import Any, Dict, List, Tuple, Union
from chart_autoencoder import models
from copy import deepcopy
import multiprocessing as mp
from functools import partial

from chart_autoencoder.get_charts import create_graph


class ModelCheckpoint:
    """Save parameters and restore them."""

    def __init__(self, path, max_to_keep=1, keep_every=1, overwrite=False):
        self.max_to_keep = max_to_keep
        self.keep_every = keep_every
        self.overwrite = overwrite

        options = ocp.CheckpointManagerOptions(
            step_prefix="checkpoint",
            max_to_keep=max_to_keep,
            save_interval_steps=keep_every,
            create=True,
        )
        checkpointers = {"params": ocp.PyTreeCheckpointer()}

        self.mngr = ocp.CheckpointManager(
            directory=path,
            options=options,
            checkpointers=checkpointers,
        )

    def save_checkpoint(self, params, step):
        self.mngr.save(step, {"params": params}, force=self.overwrite)

    def load_checkpoint(self, step):
        return self.mngr.restore(step)["params"]


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def numpy_collate_with_distances(batch):
    batch = numpy_collate(batch)
    points, idxs, distances = batch
    return points, distances[:, idxs]


def calculate_distance_matrix_single_process(chart_data, nearest_neighbors):
    pts, chart_id = chart_data
    logging.info(f"Calculating distances for chart {chart_id}")
    G = create_graph(pts=pts, nearest_neighbors=nearest_neighbors)
    # Check that graph is a single connected component
    if not nx.is_connected(G):
        raise ValueError(
            f"Graph for chart {chart_id} is not a single connected component"
        )
    distances = dict(nx.all_pairs_shortest_path_length(G, cutoff=None))
    distances_matrix = np.zeros((len(pts), len(pts)))
    for j in range(len(pts)):
        for k in range(len(pts)):
            distances_matrix[j, k] = distances[j][k]
    logging.info(f"Finished calculating distances for chart {chart_id}")
    return chart_id, distances_matrix


def compute_distance_matrix(charts, nearest_neighbors):
    """
    Compute the distance matrices for each chart.
    """
    distance_matrix = {}
    chart_data = [(charts[i], i) for i in charts.keys()]

    # Create a pool of workers
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    with mp.Pool(processes=num_processes) as pool:
        logging.info(f"Calculating distances using {num_processes} processes")
        results = pool.map(
            partial(
                calculate_distance_matrix_single_process,
                nearest_neighbors=nearest_neighbors,
            ),
            chart_data,
        )

    for chart_id, matrix in results:
        distance_matrix[chart_id] = matrix

    return distance_matrix


class BaseSampler(Dataset):
    def __init__(
        self,
        batch_size,
    ):
        self.batch_size = batch_size

    def __getitem__(self, index):
        batch = self.data_generation()
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")


class ChartSampler(BaseSampler):
    def __init__(
        self,
        charts,
        batch_size,
        distances_matrices_path=None,
        use_existing_distances_matrix=False,
        nearest_neighbors=10,
    ):
        super().__init__(batch_size)
        self.charts = charts
        self.charts_keys = charts.keys()

        self.distances_matrices = {}
        if not use_existing_distances_matrix and distances_matrices_path is not None:
            logging.info("Computing distances matrices")
            self.distances_matrices = compute_distance_matrix(charts, nearest_neighbors)
            logging.info("Saving distances matrices")
            np.save(distances_matrices_path, self.distances_matrices, allow_pickle=True)

        elif use_existing_distances_matrix and distances_matrices_path is not None:
            logging.info(f"Loading distances matrices from {distances_matrices_path}")
            self.distances_matrices = np.load(
                distances_matrices_path, allow_pickle=True
            ).item()
        else:
            logging.info("No distances matrix provided")

    def data_generation(self):
        idxs = [
            np.random.randint(0, self.charts[i].shape[0], size=(self.batch_size,))
            for i in self.charts_keys
        ]
        batch = np.stack([self.charts[i][idxs[i]] for i in self.charts_keys], axis=0)
        distances = np.stack(
            [
                self.distances_matrices[i][idxs[i], :][:, idxs[i]]
                for i in self.charts_keys
            ],
            axis=0,
        )
        return batch, distances


def prefetch_to_device(iterator, size, devices=None):
    """Prefetch batches on device.

    Note: This function is adjusted from flax.jax_utils with no sharding necessary.

    This utility takes an iterator and returns a new iterator which fills an on
    device prefetch buffer. Eager prefetching can improve the performance of
    training loops significantly by overlapping compute and data transfer.

    This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
    necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
    location that isn't free yet so they don't block. Instead those allocators OOM.

    Args:
        iterator: an iterator that yields a pytree of ndarrays where the first
        dimension is sharded across devices.

        size: the size of the prefetch buffer.

        If you're training on GPUs, 2 is generally the best choice because this
        guarantees that you can overlap a training step on GPU with a data
        prefetch step on CPU.

        devices: the list of devices to which the arrays should be prefetched.

        Defaults to the order of devices expected by `jax.pmap`.

    Yields:
        The original items from the iterator where each ndarray is now a sharded to
        the specified devices.
    """
    queue = collections.deque()
    devices = devices or jax.local_devices()

    def _prefetch(xs):
        if isinstance(xs, np.ndarray):
            return jax.device_put(
                xs, devices[0]
            )  # jax.device_put_sharded([xs], devices)
        else:
            return xs

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)


def get_model(cfg: Dict[str, Any]) -> flax.linen.Module:
    """Returns the model for the given config.

    Args:
        nef_cfg (Dict[str, Any]): The model config.

    Returns:
        flax.linen.Module: The model.

    """

    model_cfg = deepcopy(cfg).unlock()

    if model_cfg.name not in dir(models):
        raise NotImplementedError(
            f"Model {model_cfg['name']} not implemented. Available are: {dir(models)}"
        )
    else:
        model = getattr(models, model_cfg.name)
        return model(n_hidden=model_cfg.n_hidden, rff_dim=model_cfg.rff_dim)


def set_profiler(profiler_config, step, log_dir):
    if profiler_config is not None:
        if step == profiler_config.start_step:
            jax.profiler.start_trace(log_dir=log_dir)
        if step == profiler_config.end_step:
            jax.profiler.stop_trace()
