from torch.utils import data

from datasets.utils import Mesh
from pathlib import Path
import numpy as np
import os

import ml_collections

import multiprocessing as mp
from functools import partial
import networkx as nx
from sklearn.neighbors import KDTree
import scipy.linalg
from chart_autoencoder import fast_region_growing


class UniversalAEDataset(data.Dataset):
    def __init__(
        self,
        config,
        train=True,
    ):
        
        self.config = config.dataset
        self.train = train
        if self.config.create_dataset:
            m = Mesh(self.config.mesh_path)
            self.verts, self.connectivity = m.verts, m.connectivity

            points = sample_points_from_mesh(m, self.config.points_per_unit_area)
            if self.config.subset_cardinality is not None:
                if self.config.subset_cardinality < len(points):
                    rng = np.random.default_rng(self.config.seed)
                    indices = rng.choice(
                        len(points),
                        size=self.config.subset_cardinality - len(m.verts),
                        replace=False,
                    )
                    points = points[indices]

            self.points = np.concatenate([points, self.verts], axis=0)
            self._create_charts_dataset()
            self._load_charts_dataset()

        else:
            self._load_charts_dataset()
        
        self.num_points = self.charts.shape[1]
    
    def _create_charts_dataset(self):

        all_charts = []
        all_distance_matrix = []
        failed_charts = 0
        total_charts = 0
        for i in range(1, self.config.iterations+1):
            charts, charts_idxs, _ = fast_region_growing(
                pts=self.points,
                min_dist=self.config.min_dist,
                nearest_neighbors=self.config.nearest_neighbors,
            )

            total_charts += len(charts)

            normalized_charts = self._normalize_charts(charts)
            distance_matrix = compute_distance_matrix(normalized_charts, self.config.nearest_neighbors)
            for dm, chart in zip(distance_matrix, normalized_charts):
                if dm is not None:
                    all_distance_matrix = all_distance_matrix + [dm]
                    all_charts = all_charts + [chart]
                else:
                    failed_charts += 1
            
            print(f"Iteration: {i}/{self.config.iterations} ---- Failed charts: {failed_charts}/{total_charts}")

            if i%self.config.save_charts_every==0:
                Path(self.config.charts_path).mkdir(parents=True, exist_ok=True)
                np.save(self.config.charts_path + f"/charts_{i}.npy", all_charts)
                np.save(self.config.charts_path + f"/distance_matrix_{i}.npy", all_distance_matrix)
                all_charts = []
                all_distance_matrix = []

    def _load_charts_dataset(self):

        self.charts = []
        self.distance_matrix = []
        chart_files = sorted([f for f in os.listdir(self.config.charts_path) if f.startswith("charts_") and f.endswith(".npy")])
        distance_matrix_files = sorted([f for f in os.listdir(self.config.charts_path) if f.startswith("distance_matrix_") and f.endswith(".npy")])
        
        assert len(chart_files) > 0, f"No chart files found in {self.config.charts_path}"
        assert len(distance_matrix_files) > 0, f"No distance matrix files found in {self.config.charts_path}"
        assert len(chart_files) == len(distance_matrix_files), f"Unequal number of chart files ({len(chart_files)}) and distance matrix files ({len(distance_matrix_files)})"
        
        for chart_file, distance_matrix_file in zip(chart_files, distance_matrix_files):
            if self.train:
                self.charts.append(np.load(self.config.charts_path + f"/{chart_file}")[:200])
                self.distance_matrix.append(np.load(self.config.charts_path + f"/{distance_matrix_file}")[:200])
            else:
                self.charts.append(np.load(self.config.charts_path + f"/{chart_file}")[200:250])
                self.distance_matrix.append(np.load(self.config.charts_path + f"/{distance_matrix_file}")[200:250])

        self.charts = np.concatenate(self.charts, axis=0)
        self.distance_matrix = np.concatenate(self.distance_matrix, axis=0)

    def _normalize_charts(self, charts):

        normalized_charts = []
        for chart_key, chart in charts.items():
            mu = chart.mean(axis=0)
            std = chart.std()
            normalized_chart = chart-mu # /std

            random_idxs = np.random.choice(len(chart), self.config.num_points, replace=True)
            normalized_charts.append(normalized_chart[random_idxs])
        return normalized_charts

    def _get_rotated_points(self, chart_id):

        # Generate random rotation matrix
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, 2*np.pi) 
        psi = np.random.uniform(0, 2*np.pi)

        # Rotation matrix around x axis
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])

        # Rotation matrix around y axis  
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                      [0, 1, 0],
                      [-np.sin(phi), 0, np.cos(phi)]])

        # Rotation matrix around z axis
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation
        return (R @ self.charts[chart_id].T).T

    def _get_rotated_scaled_deformed_points(self, chart_id, t=0.0):
        """
        Apply rotation, scaling, and a smooth deformation to the chart points.
        
        Args:
            chart_id: index of the chart to transform
            deformation_magnitude: controls the strength of deformation (0.0 = no deformation)
        
        Returns:
            Transformed points
        """
        # Generate random rotation matrix
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, 2*np.pi) 
        psi = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.5, 1.5)

        # Rotation matrix around x axis
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])

        # Rotation matrix around y axis  
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                      [0, 1, 0],
                      [-np.sin(phi), 0, np.cos(phi)]])

        # Rotation matrix around z axis
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation and scaling
        points = (scale * R @ self.charts[chart_id].T).T
        
        # Apply a smooth, invertible deformation (diffeomorphism)

        M = np.random.normal(0, 1, (3, 3))
        TrM = np.trace(M)
        M = M - ( TrM * np.eye(3) / 3)
        
        points = (scipy.linalg.expm(t * M) @ points.T).T
        
        return points


    def __len__(self):
        return 100000 # len(self.charts)

    def __getitem__(self, idx):
        supernode_idxs = np.random.permutation(self.num_points)[: self.config.num_supernodes]
        chart_id = np.random.randint(0, len(self.charts))
        t = np.random.uniform(0, 0.2)
        points = self._get_rotated_scaled_deformed_points(chart_id, t=0.2)
        if self.config.normalize_charts:
            points = self.charts[chart_id]
            std = points.std()
            points = points/std
        else:
            points = self.charts[chart_id]
        return points, supernode_idxs, chart_id


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
    
    distances = dict(nx.all_pairs_dijkstra_path_length(G, cutoff=None))
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
    dataset.charts_path = "/scratch-shared/rvalperga/mpinns/datasets/coil/uae_dataset" # "/home/rvalperga/mpinns/datasets/coil/uae_charts"
    dataset.points_per_unit_area = 3
    dataset.subset_cardinality = None
    dataset.num_points = 500
    dataset.iterations = 30
    dataset.min_dist = 9.
    dataset.nearest_neighbors = 10
    dataset.save_charts_every = 10
    dataset.train = True
    return config


if __name__=="__main__":

    config = load_config()
    dataset = UniversalAEDataset(config=config)