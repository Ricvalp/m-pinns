import numpy as np
import jax.numpy as jnp
import multiprocessing as mp
import logging
from tqdm import tqdm
from pathlib import Path
from functools import partial
from torch.utils.data import Dataset
import networkx as nx
from sklearn.neighbors import KDTree


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
        logging.info(f"Calculating distances using {num_processes} processes")
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
        distances = dict(nx.all_pairs_shortest_path_length(G, cutoff=None))
        distances_matrix = np.zeros((len(pts), len(pts)))
        for j in range(len(pts)):
            for k in range(len(pts)):
                distances_matrix[j, k] = distances[j][k]
        return distances_matrix
    except ValueError as e:
        return None


class SphereDataset(Dataset):
    def __init__(self,
                 num_charts,
                 num_points,
                 num_supernodes,
                 nearest_neighbors_distance_matrix,
                 load_charts_and_distances=True,
                 save_charts_and_distances=True,
                 path="universal_autoencoder/sphere/data"):

        self.num_charts = num_charts
        self.num_points = num_points
        self.num_supernodes = num_supernodes
        self.nearest_neighbors_distance_matrix = nearest_neighbors_distance_matrix
        charts = []

        if load_charts_and_distances:
            self.charts = np.load(f"{path}/sphere_charts.npy")
            self.distances_matrix = np.load(f"{path}/sphere_distances_matrix.npy")
        else:
            for i in range(num_charts):
                theta = np.random.uniform(0, jnp.pi*np.random.uniform(0.3, 0.4), (num_points, 1))
                phi = np.random.uniform(0, 2 * jnp.pi, (num_points, 1))

                # Generate points on a sphere
                points = np.concatenate(
                    [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
                    axis=-1,
                )

                charts.append(points)

            distances_matrix = compute_distance_matrix(charts, self.nearest_neighbors_distance_matrix)
            self.distances_matrix = []
            self.charts = []
            invalid_charts = 0
            for i, dm in enumerate(distances_matrix):
                if dm is not None:
                    self.distances_matrix.append(dm)
                    self.charts.append(charts[i])
                else:
                    invalid_charts += 1

            print(f"Invalid charts: {invalid_charts}")

            if save_charts_and_distances:

                Path(path).mkdir(parents=True, exist_ok=True)
                np.save(f"{path}/sphere_charts.npy", self.charts)
                np.save(f"{path}/sphere_distances_matrix.npy", self.distances_matrix)

        self.random_supernode_idxs = []
        for i in range(10000):
            self.random_supernode_idxs.append(np.random.permutation(self.num_points)[: self.num_supernodes])

    def get_rotated_points(self, chart_id):

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

    def get_rotated_scaled_points(self, chart_id):

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

        # Apply rotation
        return (scale * R @ self.charts[chart_id].T).T

    def __len__(self):
        return 100000000 # self.num_points

    def __getitem__(self, idx):
        supernode_idxs = self.random_supernode_idxs[np.random.randint(0, len(self.random_supernode_idxs))]
        chart_id = np.random.randint(0, len(self.charts))
        # distance_matrix = self.distances_matrix[chart_id]
        points = self.get_rotated_scaled_points(chart_id)
        return points, supernode_idxs, chart_id  # distance_matrix


class HalfSphereDataset(Dataset):
    def __init__(self, num_points, num_supernodes, nearest_neighbors_distance_matrix):
        self.num_points = num_points
        self.num_supernodes = num_supernodes
        theta = np.random.uniform(0, jnp.pi/1.8, (num_points, 1))
        phi = np.random.uniform(0, 2 * jnp.pi, (num_points, 1))

        # Generate points on a sphere
        self.points = np.concatenate(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        )

        self.distances_matrix = calculate_distance_matrix_single_process(self.points, nearest_neighbors_distance_matrix)

        self.random_supernode_idxs = []
        for i in range(10000):
            self.random_supernode_idxs.append(np.random.permutation(self.num_points)[: self.num_supernodes])

    def get_rotated_points(self):

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
        return (R @ self.points.T).T

    def get_rotated_scaled_points(self):

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

        # Apply rotation
        return (scale * R @ self.points.T).T


    def __len__(self):
        return 100000000 # self.num_points

    def __getitem__(self, idx):
        supernode_idxs = self.random_supernode_idxs[np.random.randint(0, len(self.random_supernode_idxs))]
        # points = self.get_rotated_points()
        points = self.get_rotated_scaled_points()
        return points, supernode_idxs


if __name__ == "__main__":
    dataset = SphereDataset(num_charts=500, num_points=1000, num_supernodes=64, nearest_neighbors_distance_matrix=8)
    print(dataset[0])


