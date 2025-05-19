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


# class UniformICSampler(BaseSampler):
#     def __init__(self, x, y, u0, batch_size):
#         super().__init__(batch_size)
#         self.x = [np.array(x[key]) for key in sorted(x.keys())]
#         self.y = [np.array(y[key]) for key in sorted(y.keys())]
#         self.u0 = u0

#     def data_generation(self):
#         idxs = [
#             self.rng.integers(0, len(self.x[i]), size=(self.batch_size,))
#             for i in range(len(self.x))
#         ]

#         input_points = np.array(
#             [
#                 np.concatenate(
#                     [np.stack([self.x[i][idx], self.y[i][idx]], axis=1)],
#                     axis=1,
#                 )
#                 for i, idx in enumerate(idxs)
#             ]
#         )

#         ics = np.array([self.u0[i][idx] for i, idx in enumerate(idxs)])

#         return input_points, ics


class UniformICSampler(BaseSampler):
    def __init__(self, x, y, u0, batch_size, ics_batches_path=None):
        super().__init__(batch_size)
        self.x = [np.array(x[key]) for key in sorted(x.keys())]
        self.y = [np.array(y[key]) for key in sorted(y.keys())]
        self.u0 = u0

        if ics_batches_path is not None:
            self.ics_batches_path, self.ics_idxs_path = ics_batches_path
        else:
            self.ics_batches_path = None
            self.ics_idxs_path = None

        self.create_data_generation()

    def create_data_generation(self):

        if self.ics_batches_path is not None:
            ics_batches = np.load(self.ics_batches_path)
            ics_idxs = np.load(self.ics_idxs_path)

            def data_generation():
                idx = self.rng.integers(0, len(self.ics_batches_path), size=())
                return ics_batches[idx], ics_idxs[idx]

        else:

            def data_generation(self):
                idxs = [
                    self.rng.integers(0, len(self.x[i]), size=(self.batch_size,))
                    for i in range(len(self.x))
                ]

                input_points = np.array(
                    [
                        np.concatenate(
                            [np.stack([self.x[i][idx], self.y[i][idx]], axis=1)],
                            axis=1,
                        )
                        for i, idx in enumerate(idxs)
                    ]
                )

                ics = np.array([self.u0[i][idx] for i, idx in enumerate(idxs)])

                return input_points, ics

        self.data_generation = data_generation


class UniformSampler(BaseSampler):
    def __init__(self, x, y, T, sigma, batch_size):
        super().__init__(batch_size)
        self.x = [np.array(x[key]) for key in sorted(x.keys())]
        self.y = [np.array(y[key]) for key in sorted(y.keys())]
        self.sigma = sigma
        self.T = T

    def data_generation(self):
        idxs = [
            self.rng.integers(0, len(self.x[i]), size=(self.batch_size,))
            for i in range(len(self.x))
        ]
        batch = np.array(
            [
                np.concatenate(
                    [
                        np.stack([self.x[i][idx], self.y[i][idx]], axis=1)
                        + self.rng.normal(size=(self.batch_size, 2)) * self.sigma,
                        self.rng.uniform(size=(self.batch_size, 1)) * self.T,
                    ],
                    axis=1,
                )
                for i, idx in enumerate(idxs)
            ]
        )
        return batch


# class UniformBoundarySampler(BaseSampler):
#     def __init__(self, boundaries_x, boundaries_y, T, batch_size):
#         super().__init__(batch_size)
#         self.boundary_x = boundaries_x
#         self.boundary_y = boundaries_y
#         self.T = T
#         self.num_boundaries = min([len(boundaries_x[key]) for key in boundaries_x.keys()])

#     def data_generation(self):
#         batches = []
#         pairs_idxs = []
#         for outer_key in sorted(self.boundary_x.keys()):
#             inner_keys = self.rng.choice(
#                 np.array(list(self.boundary_x[outer_key].keys())),
#                 size=(self.num_boundaries,),
#                 replace=False,
#             )
#             cur_pairs = [(outer_key, inner_key) for inner_key in inner_keys.tolist()]
#             pairs_idxs.append(cur_pairs)

#             chart_batches = []

#             for cur_pair in cur_pairs:

#                 a = cur_pair[0]
#                 b = cur_pair[1]

#                 idx_ = self.rng.integers(
#                     0,
#                     self.boundary_x[a][b].shape[0],  # a, b
#                     size=(self.batch_size,),
#                 )

#                 _idx = self.rng.integers(
#                     0,
#                     self.boundary_x[b][a].shape[0],  # b, a
#                     size=(self.batch_size,),
#                 )

#                 T = self.rng.uniform(size=(self.batch_size, 1)) * self.T

#                 batch_ab = np.concatenate(
#                     [
#                         np.stack(
#                             [
#                                 self.boundary_x[a][b][idx_],
#                                 self.boundary_y[a][b][idx_],
#                             ],
#                             axis=1,
#                         ),
#                         T,
#                     ],
#                     axis=1,
#                 )
#                 batch_ba = np.concatenate(
#                     [
#                         np.stack(
#                             [
#                                 self.boundary_x[b][a][_idx],
#                                 self.boundary_y[b][a][_idx],
#                             ],
#                             axis=1,
#                         ),
#                         T,
#                     ],
#                     axis=1,
#                 )

#                 batch = np.stack([batch_ab, batch_ba], axis=0)
#                 chart_batches.append(batch)
#             batches.append(chart_batches)

#         return np.array(batches), np.array(pairs_idxs)


class UniformBoundarySampler(BaseSampler):
    def __init__(
        self, boundaries_x, boundaries_y, T, batch_size, boundary_batches_paths=None
    ):
        super().__init__(batch_size)
        self.boundary_x = boundaries_x
        self.boundary_y = boundaries_y
        self.T = T
        self.num_boundaries = min(
            [len(boundaries_x[key]) for key in boundaries_x.keys()]
        )

        if boundary_batches_paths is not None:
            self.boundary_batches_path, self.boundary_pairs_idxs_path = (
                boundary_batches_paths
            )
        else:
            self.boundary_batches_path = None
            self.boundary_pairs_idxs_path = None

        self.create_data_generation()

    def create_data_generation(self):

        if self.boundary_batches_path is not None:
            boundary_batches = np.load(self.boundary_batches_path)
            boundary_pairs_idxs = np.load(self.boundary_pairs_idxs_path)

            def data_generation():
                idx = self.rng.integers(0, boundary_batches.shape[0], size=())
                return boundary_batches[idx], boundary_pairs_idxs[idx]

        else:

            def data_generation():
                batches = []
                pairs_idxs = []
                for outer_key in sorted(self.boundary_x.keys()):
                    inner_keys = self.rng.choice(
                        np.array(list(self.boundary_x[outer_key].keys())),
                        size=(self.num_boundaries,),
                        replace=False,
                    )
                    cur_pairs = [
                        (outer_key, inner_key) for inner_key in inner_keys.tolist()
                    ]
                    pairs_idxs.append(cur_pairs)

                    chart_batches = []

                    for cur_pair in cur_pairs:

                        a = cur_pair[0]
                        b = cur_pair[1]

                        idx_ = self.rng.integers(
                            0,
                            self.boundary_x[a][b].shape[0],  # a, b
                            size=(self.batch_size,),
                        )

                        _idx = self.rng.integers(
                            0,
                            self.boundary_x[b][a].shape[0],  # b, a
                            size=(self.batch_size,),
                        )

                        T = self.rng.uniform(size=(self.batch_size, 1)) * self.T

                        batch_ab = np.concatenate(
                            [
                                np.stack(
                                    [
                                        self.boundary_x[a][b][idx_],
                                        self.boundary_y[a][b][idx_],
                                    ],
                                    axis=1,
                                ),
                                T,
                            ],
                            axis=1,
                        )
                        batch_ba = np.concatenate(
                            [
                                np.stack(
                                    [
                                        self.boundary_x[b][a][_idx],
                                        self.boundary_y[b][a][_idx],
                                    ],
                                    axis=1,
                                ),
                                T,
                            ],
                            axis=1,
                        )

                        batch = np.stack([batch_ab, batch_ba], axis=0)
                        chart_batches.append(batch)
                    batches.append(chart_batches)

                return np.array(batches), np.array(pairs_idxs)

        self.data_generation = data_generation
