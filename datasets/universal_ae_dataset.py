from torch.utils import data
from typing import Any, Dict, List, Tuple, Union
from datasets.utils import Mesh
from pathlib import Path
import numpy as np
import os

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
            m = Mesh(self.config.mesh_path)
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
        failed_charts = 0
        for i in range(self.config.iterations):
            charts, charts_idxs, _ = fast_region_growing(
                pts=self.points,
                min_dist=self.config.min_dist,
                nearest_neighbors=self.config.nearest_neighbors,
            )

            normalized_charts = self._normalize_charts(charts)
            distance_matrix = compute_distance_matrix(normalized_charts, self.config.nearest_neighbors)
            for dm, chart in zip(distance_matrix, normalized_charts):
                if dm is not None:
                    self.distance_matrix = self.distance_matrix + [dm]
                    self.charts = self.charts + [chart]
                else:
                    failed_charts += 1
            
            print(f"Failed charts: {failed_charts}/{len(distance_matrix)}")

            if i%self.config.save_charts_every==0:
                Path(self.config.charts_path).mkdir(parents=True, exist_ok=True)
                np.save(self.config.charts_path + f"/charts_{i}.npy", self.charts)
                np.save(self.config.charts_path + f"/distance_matrix_{i}.npy", self.distance_matrix)


    def load_charts_dataset(self):

        self.charts = []
        self.distance_matrix = []
        # List all chart files in the directory
        chart_files = sorted([f for f in os.listdir(self.config.charts_path) if f.startswith("charts_") and f.endswith(".npy")])
        distance_matrix_files = sorted([f for f in os.listdir(self.config.charts_path) if f.startswith("distance_matrix_") and f.endswith(".npy")])
        
        assert len(chart_files) > 0, f"No chart files found in {self.config.charts_path}"
        assert len(distance_matrix_files) > 0, f"No distance matrix files found in {self.config.charts_path}"
        assert len(chart_files) == len(distance_matrix_files), f"Unequal number of chart files ({len(chart_files)}) and distance matrix files ({len(distance_matrix_files)})"
        
        for i in range(len(chart_files)):
            self.charts.append(np.load(self.config.charts_path + f"/charts_{i}.npy"))
            self.distance_matrix.append(np.load(self.config.charts_path + f"/distance_matrix_{i}.npy"))

        self.charts = np.concatenate(self.charts, axis=0)
        self.distance_matrix = np.concatenate(self.distance_matrix, axis=0)

    
    def _normalize_charts(self, charts):

        normalized_charts = []
        for chart_key, chart in charts.items():
            mu = chart.mean(axis=0)
            std = chart.std(axis=0)

            random_idxs = np.random.choice(len(chart), self.config.num_points, replace=True)
            normalized_chart = (chart-mu)/std
            normalized_charts.append(normalized_chart[random_idxs])

        return normalized_charts

    def __len__(self):
        return 100000 # len(self.charts)

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.charts))
        return self.charts[i], self.distance_matrix[i]


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
    try:
        if not nx.is_connected(G):
            raise ValueError(
                f"Graph for chart {chart_id} is not a single connected component"
            )
    except Exception as e:
        print(f"Error creating graph for chart {chart_id}: {e}")
        return None
    
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
    dataset.seed = 37
    dataset.create_dataset = True
    dataset.mesh_path = "./datasets/coil/coil_1.2_MM.obj"
    dataset.charts_path = "/home/rvalperga/mpinns/datasets/coil/uae_charts"
    dataset.points_per_unit_area = 3
    dataset.subset_cardinality = 100000
    dataset.num_points = 1000
    dataset.iterations = 100
    dataset.min_dist = 10.
    dataset.nearest_neighbors = 10
    dataset.save_charts_every = 10

    return config


if __name__=="__main__":

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    config = load_config()
    dataset = UniversalAEDataset(config=config)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in tqdm(data_loader):
        assert True