from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from torch.utils.data import Dataset


class BaseSampler(Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(1234)

    def __getitem__(self, index):
        "Generate one batch of data"
        batch = self.data_generation()
        return batch

    def data_generation(self):
        raise NotImplementedError("Subclasses should implement this!")


class UniformICSampler(BaseSampler):
    def __init__(self, x, y, u0, batch_size):
        super().__init__(batch_size)
        self.x = x
        self.y = y
        self.u0 = u0

        self.create_data_generation()

    def create_data_generation(self):

        def data_generation():
            idxs = self.rng.integers(0, len(self.x), size=(self.batch_size,))

            input_points = np.stack([self.x[idxs], self.y[idxs]], axis=1)

            ics = self.u0[idxs]

            return input_points, ics

        self.data_generation = data_generation


class UniformSampler(BaseSampler):
    def __init__(self, x, y, ts, sigma, batch_size):
        super().__init__(batch_size)
        self.x = x
        self.y = y
        self.ts = ts
        self.sigma = sigma

    def data_generation(self):
        idxs = self.rng.integers(0, len(self.x), size=(self.batch_size,))
        ts_idxs = self.rng.integers(0, len(self.ts), size=(self.batch_size,))
        batch = np.stack([self.x[idxs], self.y[idxs], self.ts[ts_idxs]], axis=1)
        return batch, ts_idxs
