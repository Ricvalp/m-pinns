import numpy as np
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


class UniformBCSampler(BaseSampler):
    def __init__(
        self,
        bcs_x,
        bcs_y,
        bcs,
        batch_size,
        num_charts,
        bcs_batches_path=None,
        load_existing_batches=False,
    ):
        super().__init__(batch_size)
        self.bcs_x = [np.array(bcs_x[key]) for key in sorted(bcs_x.keys())]
        self.bcs_y = [np.array(bcs_y[key]) for key in sorted(bcs_y.keys())]
        self.bcs = [np.array(bcs[key]) for key in sorted(bcs.keys())]

        self.bcs_x = [
            np.array(bcs_x[key]) if key in bcs_x.keys() else np.array([0.0])
            for key in range(num_charts)
        ]
        self.bcs_y = [
            np.array(bcs_y[key]) if key in bcs_y.keys() else np.array([0.0])
            for key in range(num_charts)
        ]
        self.bcs = [
            np.array(bcs[key]) if key in bcs.keys() else np.array([0.0])
            for key in range(num_charts)
        ]

        self.load_existing_batches = load_existing_batches
        if load_existing_batches:
            self.bcs_batches_path, self.bcs_values_path = bcs_batches_path

        self.create_data_generation()

    def create_data_generation(self):

        if self.load_existing_batches:
            self.bcs_batches = np.load(self.bcs_batches_path)
            self.bcs_values = np.load(self.bcs_values_path)

            def data_generation():
                idx = self.rng.integers(0, len(self.bcs_batches_path), size=())
                return self.bcs_batches[idx], self.bcs_values[idx]

        else:

            def data_generation():
                idxs = [
                    self.rng.integers(0, len(self.bcs_x[i]), size=(self.batch_size,))
                    for i in range(len(self.bcs_x))
                ]

                input_points = np.array(
                    [
                        np.concatenate(
                            [
                                np.stack(
                                    [self.bcs_x[i][idx], self.bcs_y[i][idx]], axis=1
                                )
                            ],
                            axis=1,
                        )
                        for i, idx in enumerate(idxs)
                    ]
                )

                bcs = np.array([self.bcs[i][idx] for i, idx in enumerate(idxs)])

                return input_points, bcs

        self.data_generation = data_generation


class UniformSampler(BaseSampler):
    def __init__(self, x, y, sigma, batch_size):
        super().__init__(batch_size)
        self.x = [np.array(x[key]) for key in sorted(x.keys())]
        self.y = [np.array(y[key]) for key in sorted(y.keys())]
        self.sigma = sigma

    def data_generation(self):
        idxs = [
            self.rng.integers(0, len(self.x[i]), size=(self.batch_size,))
            for i in range(len(self.x))
        ]
        batch = np.array(
            [
                np.stack([self.x[i][idx], self.y[i][idx]], axis=1)
                + self.rng.normal(size=(self.batch_size, 2)) * self.sigma
                for i, idx in enumerate(idxs)
            ]
        )
        return batch


class UniformBoundarySampler(BaseSampler):
    def __init__(
        self,
        boundaries_x,
        boundaries_y,
        batch_size,
        boundary_batches_paths=None,
        load_existing_batches=False,
    ):
        super().__init__(batch_size)
        self.boundary_x = boundaries_x
        self.boundary_y = boundaries_y
        self.num_boundaries = min(
            [len(boundaries_x[key]) for key in boundaries_x.keys()]
        )

        self.load_existing_batches = load_existing_batches
        if load_existing_batches:
            self.boundary_batches_path, self.boundary_pairs_idxs_path = (
                boundary_batches_paths
            )

        self.create_data_generation()

    def create_data_generation(self):

        if self.load_existing_batches:
            self.boundary_batches = np.load(self.boundary_batches_path)
            self.boundary_pairs_idxs = np.load(self.boundary_pairs_idxs_path)

            def data_generation():
                idx = self.rng.integers(0, self.boundary_batches.shape[0], size=())
                return self.boundary_batches[idx], self.boundary_pairs_idxs[idx]

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

                        batch_ab = np.stack(
                            [
                                self.boundary_x[a][b][idx_],
                                self.boundary_y[a][b][idx_],
                            ],
                            axis=1,
                        )
                        batch_ba = np.stack(
                            [
                                self.boundary_x[b][a][_idx],
                                self.boundary_y[b][a][_idx],
                            ],
                            axis=1,
                        )

                        batch = np.stack([batch_ab, batch_ba], axis=0)
                        chart_batches.append(batch)
                    batches.append(chart_batches)

                return np.array(batches), np.array(pairs_idxs)

        self.data_generation = data_generation
