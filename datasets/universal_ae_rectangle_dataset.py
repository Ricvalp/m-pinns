from torch.utils import data
import matplotlib.pyplot as plt

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
import torch
import math



class UniversalAERectangleDataset(data.Dataset):
    def __init__(
        self,
        config,
        train=True,
    ):
        
        self.config = config.dataset
        self.train = train
        
        if self.config.create_dataset:
            x = np.linspace(0, 1, 40)
            y = np.linspace(0, 1, 40)
            xx, yy = np.meshgrid(x, y)
            points = np.zeros((xx.size, 3))
            points[:, 0] = xx.flatten()
            points[:, 1] = yy.flatten()
            points[:, 2] = 0.0
            self.grid = points
            self.distance_matrix = calculate_distance_matrix_single_process(
                (self.grid, 0),
                self.config.nearest_neighbors
            )
            self._create_charts_dataset()
            self._load_charts_dataset()

        else:
            self._load_charts_dataset()
        
        self.num_points = self.charts.shape[1]
    

    def _create_charts_dataset(self):

        all_charts = []
        all_distance_matrix = []
        for i in range(1, self.config.iterations+1):
            idxs = np.random.permutation(len(self.grid))[:self.config.num_points]
            points = self.grid[idxs]
            distance_matrix = self.distance_matrix[idxs, :][:, idxs]
            all_charts.append(points)
            all_distance_matrix.append(distance_matrix)
       
        Path(self.config.charts_path).mkdir(parents=True, exist_ok=True)
        np.save(self.config.charts_path + f"/charts.npy", all_charts)
        np.save(self.config.charts_path + f"/distance_matrix.npy", all_distance_matrix)

    def _load_charts_dataset(self):

        if self.train:
            self.charts = np.load(self.config.charts_path + f"/charts.npy")[:100]
            self.distance_matrix = np.load(self.config.charts_path + f"/distance_matrix.npy")[:100]
        else:
            self.charts = np.load(self.config.charts_path + f"/charts.npy")[100:150]
            self.distance_matrix = np.load(self.config.charts_path + f"/distance_matrix.npy")[100:150]


    def _get_deformed_points(self, chart_id, t):
        """
        Apply a smooth deformation in the z-direction to the chart points.
        
        Args:
            chart_id: index of the chart to transform
            t: controls the strength of deformation (0.0 = no deformation)
        
        Returns:
            Transformed points
        """
        # Get the base points for this chart
        points = self.charts[chart_id].copy()  # Make a copy to avoid modifying original data
        
        # Get x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        # Generate random frequencies (but still zero at boundaries)
        freq_x = np.random.randint(1, 4, size=(2,))  # Random integer between 1 and 3
        freq_y = np.random.randint(1, 4, size=(2,))
        
        # Create a 2D sine wave deformation in z-direction that is zero at the boundaries
        # Sum over various frequencies for a more complex deformation pattern
        deformation_z = np.zeros_like(x)
        for i in range(len(freq_x)):
            # Each term is zero when x=0, x=1, y=0, or y=1
            deformation_z += t[i] * np.sin(freq_x[i] * np.pi * x) * np.sin(freq_y[i] * np.pi * y)
        
        # Apply the deformation to z coordinate
        points[:, 2] = deformation_z
        
        return points

    def __len__(self):
        return 100000 # len(self.charts)

    def __getitem__(self, idx):

        supernode_idxs = np.random.permutation(self.num_points)[: self.config.num_supernodes]
        chart_id = np.random.randint(0, len(self.charts))
        t = np.random.uniform(0, .8, size=(4,))
        points = self._get_deformed_points(chart_id, t=t)
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


def plot_dataset(chart, supernode_idxs, distance_matrix, name=None):
    """
    Plot a batch of charts with the supernodes highlighted in a different color.
    
    Args:
        chart: Batch of 3D points representing charts
        supernode_idxs: Indices of supernodes for each chart
        distance_matrix: Distance matrix of the charts
        name: Optional path to save the figure
    """
    # Determine grid layout
    num_samples = min(16, len(chart))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    # Compute global min/max across all charts for consistent scaling
    x_min_global = np.min([chart[i][:, 0].min() for i in range(num_samples)])
    x_max_global = np.max([chart[i][:, 0].max() for i in range(num_samples)])
    y_min_global = np.min([chart[i][:, 1].min() for i in range(num_samples)])
    y_max_global = np.max([chart[i][:, 1].max() for i in range(num_samples)])
    z_min_global = np.min([chart[i][:, 2].min() for i in range(num_samples)])
    z_max_global = np.max([chart[i][:, 2].max() for i in range(num_samples)])
    
    # Add 10% padding to the global ranges
    x_pad = 0.1 * (x_max_global - x_min_global)
    y_pad = 0.1 * (y_max_global - y_min_global)
    z_pad = 0.1 * (z_max_global - z_min_global)
    
    # Compute global limits
    x_limits = [x_min_global - x_pad, x_max_global + x_pad]
    y_limits = [y_min_global - y_pad, y_max_global + y_pad]
    z_limits = [z_min_global - z_pad, z_max_global + z_pad]
    
    # Find global min/max for colormap scaling (distances)
    all_distances = np.concatenate([distance_matrix[i][0] for i in range(num_samples)])
    distance_min = all_distances.min()
    distance_max = all_distances.max()
    
    fig = plt.figure(figsize=(12, num_samples))
    
    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Create a mask for supernodes
        is_supernode = np.zeros(len(chart[i]), dtype=bool)
        is_supernode[supernode_idxs[i]] = True

        distances = distance_matrix[i]
        
        # Plot regular points with consistent colormap scaling
        ax.scatter(
            chart[i][~is_supernode, 0],
            chart[i][~is_supernode, 1],
            chart[i][~is_supernode, 2],
            c=distances[0][~is_supernode],
            alpha=0.5,
            s=10,
            label='Regular Points',
            vmin=distance_min,
            vmax=distance_max
        )
        
        # Plot supernodes with different color
        ax.scatter(
            chart[i][is_supernode, 0],
            chart[i][is_supernode, 1],
            chart[i][is_supernode, 2],
            c='red',
            alpha=1.0,
            s=30,
            label='Supernodes'
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Use global limits for all plots
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.legend()
    
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, dpi=300)
    plt.close()


def load_config():

    config = ml_collections.ConfigDict()
    
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.seed = 37
    dataset.create_dataset = False # True
    dataset.charts_path = "/scratch-shared/rvalperga/mpinns/datasets/rectangle/"
    dataset.nearest_neighbors = 4
    dataset.num_supernodes = 32
    dataset.num_points = 1000
    dataset.iterations = 150
    dataset.train = True
    return config


if __name__=="__main__":

    config = load_config()
    dataset = UniversalAERectangleDataset(config=config)
    distance_matrix = dataset.distance_matrix
    data_loader = data.DataLoader(dataset, batch_size=16, shuffle=True)
    exmp_chart, exmp_supernode_idxs, exmp_chart_id = next(iter(data_loader))
    plot_dataset(exmp_chart, exmp_supernode_idxs, distance_matrix[exmp_chart_id], name="./figures/rectangle_dataset_with_supernodes.png")