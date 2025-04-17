from datasets.utils import Mesh
from torch.utils import data
import numpy as np
import igl
from jax import random


class Coil(data.Dataset):
    def __init__(
        self,
        path="./datasets/coil/coil_1.2_MM.obj",
        scale=0.1,
        points_per_unit_area=None,
        subset_cardinality=None,
        seed=42,
    ):

        m = Mesh(path)
        self.verts, self.connectivity = m.verts * scale, m.connectivity

        self.data = sample_points_from_mesh(m, points_per_unit_area) * scale
        if subset_cardinality is not None:
            rng = np.random.default_rng(seed)
            if subset_cardinality < len(self.data):
                indices = rng.choice(
                    len(self.data),
                    size=subset_cardinality - len(m.verts),
                    replace=False,
                )
                self.data = self.data[indices]

        self.data = np.concatenate([self.data, self.verts], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def sample_points_from_mesh(m, points_per_unit_area=2):
    """
    Sample points from the mesh triangles.

    Args:
        bunny_mesh (mesh.Mesh): Mesh object
        points_per_unit_area (float): Number of points to sample per unit area

    Returns:
        numpy.ndarray: Point cloud array with shape (N, 3)
    """
    all_points = []

    for triangle in m.connectivity:
        # Get triangle vertices
        v1, v2, v3 = m.verts[triangle]

        # Calculate triangle area using cross product
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal) / 2

        # Calculate number of points to sample based on area
        num_samples = max(1, int(area * points_per_unit_area))

        # Generate random barycentric coordinates
        r1 = np.random.random((num_samples, 1))
        r2 = np.random.random((num_samples, 1))

        # Ensure the random points lie within the triangle
        mask = (r1 + r2) > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]

        # Calculate barycentric coordinates
        a = 1 - r1 - r2
        b = r1
        c = r2

        # Generate points using barycentric coordinates
        points = a * v1 + b * v2 + c * v3
        all_points.append(points)

    # Combine all points into single array
    point_cloud = np.vstack(all_points)

    return point_cloud
