from torch.utils import data
from typing import Any, Dict, List, Tuple, Union
from datasets.utils import Mesh
from pathlib import Path
import numpy as np

import ml_collections

import multiprocessing as mp
from functools import partial
import networkx as nx
from sklearn.neighbors import KDTree

from chart_autoencoder import fast_region_growing


class UniversalAEDataset(data.Dataset):
    def __init__(
        self,
        config,
    ):
        
        self.config = config.dataset
        if self.config.create_dataset:
            m = Mesh(self.config.path)
            self.verts, self.connectivity = m.verts, m.connectivity

            points = sample_points_from_mesh(m, self.config.points_per_unit_area)
            if self.config.subset_cardinality is not None:
                rng = np.random.default_rng(self.config.seed)
                if self.config.subset_cardinality < len(points):
                    indices = rng.choice(
                        len(points),
                        size=self.config.subset_cardinality - len(m.verts),
                        replace=False,
                    )
                    points = points[indices]

            self.points = np.concatenate([points, self.verts], axis=0)
            self.create_charts_dataset()
        
        else:
            self.load_charts_dataset()
    
    def create_charts_dataset(self):

        self.charts = []
        self.distance_matrix = []
        for i in range(self.config.num_charts):
            charts, charts_idxs, _ = fast_region_growing(
                pts=self.points,
                min_dist=self.config.min_dist,
                nearest_neighbors=self.config.nearest_neighbors,
            )

            normalized_charts = self._normalize_charts(charts)
            self.charts = self.charts + normalized_charts

            distance_matrix = compute_distance_matrix(normalized_charts, self.nearest_neighbors_distance_matrix)
            self.distance_matrix = self.distance_matrix + distance_matrix
        
        Path(self.config.path).mkdir(parents=True, exist_ok=True)
        np.save(self.config.path + "/charts.npy", self.charts)
        np.save(self.config.path + "/distance_matrix.npy", self.distance_matrix)


    def load_charts_dataset(self):

        self.charts = np.load(self.config.path)
        self.distance_matrix = np.load(self.config.path)

    
    def _normalize_charts(self, charts):

        normalized_charts = []
        for chart in charts:
            mu = chart.mean()
            std = chart.std()

            random_idxs = np.random.choice(len(chart), self.config.num_points, replace=True)
            normalized_chart = (charts-mu)/std
            normalized_charts.append(normalized_chart[random_idxs])

        return normalized_charts


def create_graph(
    pts,
    nearest_neighbors,
):
    """

    Create a graph from the points.

    Args:
        pts (np.ndarray): The points
        n (int): The number of nearest neighbors
        connectivity (np.ndarray): The connectivity of the mesh

    Returns:
        G (nx.Graph): The graph

    """

    # Create a n-NN graph
    tree = KDTree(pts)
    G = nx.Graph()

    # Add nodes to the graph
    for i, point in enumerate(pts):
        G.add_node(i, pos=point)

    # Add edges to the graph
    distances, indices = tree.query(
        pts, nearest_neighbors + 1
    )  # n+1 because the point itself is included

    for i in range(len(pts)):
        for j in range(
            1, nearest_neighbors + 1
        ):  # start from 1 to exclude the point itself
            neighbor_index = indices[i, j]
            distance = distances[i, j]
            G.add_edge(i, neighbor_index, weight=distance)

    return G


def compute_distance_matrix(charts, nearest_neighbors):
    """
    Compute the distance matrices for each chart.
    """
    chart_data = [(chart, i) for i, chart in enumerate(charts)]

    # Create a pool of workers
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    with mp.Pool(processes=num_processes) as pool:
        distance_matrices = pool.map(
            partial(
                calculate_distance_matrix_single_process,
                nearest_neighbors=nearest_neighbors,
            ),
            chart_data,
        )

    return distance_matrices


def calculate_distance_matrix_single_process(chart_data, nearest_neighbors):
    pts, chart_id = chart_data
    G = create_graph(pts=pts, nearest_neighbors=nearest_neighbors)
    if not nx.is_connected(G):
        raise ValueError(
            f"Graph for chart {chart_id} is not a single connected component"
        )
    distances = dict(nx.all_pairs_shortest_path_length(G, cutoff=None))
    distances_matrix = np.zeros((len(pts), len(pts)))
    for j in range(len(pts)):
        for k in range(len(pts)):
            distances_matrix[j, k] = distances[j][k]
    return distances_matrix


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


def load_config():

    config = ml_collections.ConfigDict()
    
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.create_dataset = True
    dataset.path = "./datasets/coil/coil_1.2_MM.obj"
    dataset.points_per_unit_area = 3
    dataset.subset_cardinality = 100000
    dataset.num_points = 1000
    dataset.num_charts = 200
    dataset.min_dist = 1.
    dataset.nearest_neighbors = 8


if __name__=="__main__":

    config = load_config()
    dataset = UniversalAEDataset(config=config)
    
    print(dataset[0].shape)