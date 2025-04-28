import numpy as np


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
