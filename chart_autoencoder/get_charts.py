import numpy as np

from scipy.spatial import KDTree
from tqdm import tqdm
import logging
from typing import Any, Dict, List, Tuple, Union
import pickle


from collections import deque
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.spatial import KDTree


import networkx as nx
import jax.numpy as jnp

from pathlib import Path
from collections import deque


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def geodesic_distance(G, point_index_1, point_index_2):
    return nx.shortest_path_length(
        G, source=point_index_1, target=point_index_2, weight="weight"
    )


def calc_geodesic_distances(G, point_index_1, points):
    distances = np.zeros(len(points))
    for i in range(len(points)):
        distances[i] = geodesic_distance(G, point_index_1, i)
    return distances


def get_n_hop_neighbors(G, node, nn):
    # Find all nodes within n hops
    hop_neighbors = [
        neighbor
        for neighbor, length in nx.single_source_shortest_path_length(G, node).items()
        if length <= nn
    ]
    return hop_neighbors


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
    logging.info("Building the graph...")
    for i, point in enumerate(pts):
        distances, indices = tree.query(
            point, nearest_neighbors + 1
        )  # n+1 because the point itself is included
        for j in range(
            1, nearest_neighbors + 1
        ):  # start from 1 to exclude the point itself
            neighbor_index = indices[j]
            distance = distances[j]
            G.add_edge(i, neighbor_index, weight=distance)

    logging.info(f"Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges")

    return G


def create_graph_from_mesh(verts, connectivity):
    G = nx.Graph()

    # Add nodes to the graph
    for i, point in enumerate(verts):
        G.add_node(i, pos=point)

    # Add edges based on the mesh connectivity
    logging.info("Building the graph from mesh connectivity...")
    with tqdm(total=len(connectivity)) as pbar:
        for triangle in connectivity:
            # Each triangle is formed by three vertices
            for i in range(3):
                for j in range(i + 1, 3):
                    v1 = triangle[i]
                    v2 = triangle[j]
                    distance = np.linalg.norm(verts[v1] - verts[v2])
                    G.add_edge(v1, v2, weight=distance)
            pbar.update(1)

    return G


def poisson_disk_sampling(points, min_dist):
    """
    Perform Poisson disk sampling on a 3D graph.

    Args:
        points (np.array): Nx3 array of 3D points.
        graph (nx.Graph): Graph connecting nearest neighbors.
        min_dist (float): Minimum allowed distance between sampled points.

    Returns:
        sampled_points (np.array): Poisson-disk sampled subset of points.
    """
    tree = cKDTree(points)
    sampled = []
    unprocessed = set(range(len(points)))

    logging.info(f"Poisson disk sampling {len(points)} points")

    while unprocessed:
        idx = np.random.choice(list(unprocessed))
        sampled.append(idx)
        unprocessed.remove(idx)

        # Remove points that are within min_dist
        neighbors = tree.query_ball_point(points[idx], min_dist)
        unprocessed.difference_update(neighbors)
        if len(sampled) % 5 == 0:
            logging.info(f"Total seed points: {len(sampled)}")

    logging.info(f"Poisson disk sampled {len(sampled)} points")

    return sampled


def fast_region_growing(
    pts, min_dist, nearest_neighbors=5, original_idxs=None
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Fast region growing algorithm.

    Args:
        pts (np.ndarray): The points
        min_dist (float): The minimum distance between points
        nearest_neighbors (int): The number of nearest neighbors

    Returns:
        partitions (Dict[int, np.ndarray]): The partitions
        seed_points (np.ndarray): The seed points
    """

    logging.info(
        f"Creating graph with fast region growing: min_dist={min_dist}, n={nearest_neighbors}"
    )
    if original_idxs is None:
        original_idxs = np.arange(len(pts))

    G = create_graph(
        pts=pts,
        nearest_neighbors=nearest_neighbors,
    )

    seed_points = poisson_disk_sampling(pts, min_dist)

    logging.info(f"Starting region growing")

    partitions = {}
    queue = deque()

    # Initialize the queue with seed points and assign them to partitions

    for label, seed_point in enumerate(seed_points):
        partitions[seed_point] = label
        queue.append((seed_point, label))

    with tqdm(total=len(G.nodes), desc="Region Growing") as pbar:
        while queue:
            node, label = queue.popleft()
            for neighbor in G.neighbors(node):
                if neighbor not in partitions:
                    partitions[neighbor] = label
                    queue.append((neighbor, label))
            pbar.update(1)

    partitions_idxs = {
        i: [point for point in partitions if partitions[point] == i]
        for i in range(len(seed_points))
    }

    nodes = list(partitions.keys())
    for node in tqdm(nodes, desc="Adding one-hop overlap"):
        node_partition_idx = partitions[node]
        for neighbor in G.neighbors(node):
            if neighbor not in partitions_idxs[node_partition_idx]:
                partitions_idxs[node_partition_idx].append(neighbor)

    partitions = {i: pts[partitions_idxs[i]] for i in partitions_idxs.keys()}
    partitions_idxs = {
        i: original_idxs[partitions_idxs[i]] for i in partitions_idxs.keys()
    }

    logging.info(f"Checking if the union of partitions covers all points")

    all_partition_points = set()
    for partition_points in partitions.values():
        all_partition_points.update([tuple(p) for p in partition_points])

    all_input_points = set([tuple(p) for p in pts])

    if len(all_partition_points) != len(all_input_points):
        logging.warning(
            f"Partitions only cover {len(all_partition_points)} points out of {len(all_input_points)} total points"
        )
        missing_points = all_input_points - all_partition_points
        logging.warning(f"Missing {len(missing_points)} points")

    else:
        logging.info(
            f"Partitioning completed: all {len(all_partition_points)} points are covered by the partitions"
        )

    return partitions, partitions_idxs, pts[seed_points]


def refine_chart(
    points: jnp.ndarray, points_idxs, charts_to_refine_cfg: Dict[str, Any]
) -> List[jnp.ndarray]:
    """
    Returns the charts from a mesh

    Args:
        points (jnp.ndarray): The mesh points
        cfg (Dict[str, Any]): The configuration
        connectivity (jnp.ndarray): The connectivity of the mesh
    """

    charts, partitions_idxs, sampled_points = fast_region_growing(
        pts=points,
        min_dist=charts_to_refine_cfg.min_dist,
        nearest_neighbors=charts_to_refine_cfg.nearest_neighbors,
        original_idxs=points_idxs,
    )

    return charts, partitions_idxs, sampled_points


def reindex_charts(
    old_charts: List[jnp.ndarray],
    old_idxs: Dict[int, List[int]],
    key_chart_to_refine: int,
    refined_charts: List[jnp.ndarray],
    refined_idxs: Dict[int, List[int]],
) -> List[jnp.ndarray]:
    """
    Reindex the charts and boundaries after refining a chart
    """

    old_charts.pop(key_chart_to_refine)
    old_charts[key_chart_to_refine] = refined_charts[0]

    len_old_charts = len(old_charts)
    for key in refined_charts.keys():
        if key != 0:
            old_charts[key + len_old_charts - 1] = refined_charts[key]

    old_idxs.pop(key_chart_to_refine)
    old_idxs[key_chart_to_refine] = refined_idxs[0]

    len_old_idxs = len(old_idxs)
    for key in refined_idxs.keys():
        if key != 0:
            old_idxs[key + len_old_idxs - 1] = refined_idxs[key]

    boundaries, boundary_indices = get_boundaries(old_charts)

    return old_charts, old_idxs, boundaries, boundary_indices


def get_boundaries(charts: List[jnp.ndarray]) -> Dict[Tuple[int, int], jnp.ndarray]:
    """
    Returns the boundaries between the charts as 3D points

    Args:
        charts: List of arrays containing 3D points for each chart

    Returns:
        Dictionary mapping pairs of chart indices to arrays of boundary points
    """
    boundaries = {}
    boundary_indices = {}

    for i in tqdm(range(len(charts)), desc="Getting boundaries"):
        for j in range(i + 1, len(charts)):
            points_i = set(map(tuple, charts[i]))
            points_j = set(map(tuple, charts[j]))

            boundary_points = points_i.intersection(points_j)

            if len(boundary_points) > 0:
                boundaries[(i, j)] = np.array(list(boundary_points))

                indices_i = [
                    list(map(tuple, charts[i])).index(p) for p in boundary_points
                ]
                indices_j = [
                    list(map(tuple, charts[j])).index(p) for p in boundary_points
                ]
                boundary_indices[(i, j)] = indices_i
                boundary_indices[(j, i)] = indices_j

    return boundaries, boundary_indices


def load_charts(
    charts_path: Union[str, Path],
) -> Tuple[
    List[jnp.ndarray],
    Dict[Tuple[int, int], jnp.ndarray],
    Dict[Tuple[int, int], Tuple[List[int], List[int]]],
]:
    """Load charts from a folder.

    Args:
        charts_path (Union[str, Path]): The path to the folder where the charts are stored.
        from_autodecoder (bool): Whether to load 2D charts as well.

    Returns:
        List[np.ndarray]: The loaded charts.
        Dict[Tuple[int, int], jnp.ndarray]: The loaded boundaries.
        Dict[Tuple[int, int], Tuple[List[int], List[int]]]: The loaded boundary indices.
        Dict[int, jnp.ndarray]: The loaded 2D charts.
    """

    logging.info(f"Loading charts and boundaries from {charts_path}...")

    with open(charts_path + "/charts.pkl", "rb") as f:
        loaded_charts = pickle.load(f)

    with open(charts_path + "/boundaries.pkl", "rb") as f:
        loaded_boundaries = pickle.load(f)

    with open(charts_path + "/boundary_indices.pkl", "rb") as f:
        loaded_boundary_indices = pickle.load(f)
    with open(charts_path + "/charts_idxs.pkl", "rb") as f:
        loaded_charts_idxs = pickle.load(f)
    try:
        with open(charts_path + "/charts2d.pkl", "rb") as f:
            loaded_charts2d = pickle.load(f)
    except:
        logging.info("No 2D charts found")
        loaded_charts2d = None

    return (
        loaded_charts,
        loaded_charts_idxs,
        loaded_boundaries,
        loaded_boundary_indices,
        loaded_charts2d,
    )



def load_charts3d(
    charts_path: Union[str, Path],
) -> Tuple[
    List[jnp.ndarray],
    Dict[Tuple[int, int], jnp.ndarray],
    Dict[Tuple[int, int], Tuple[List[int], List[int]]],
]:
    """Load charts from a folder.

    Args:
        charts_path (Union[str, Path]): The path to the folder where the charts are stored.

    Returns:
        List[np.ndarray]: The loaded charts.
        Dict[Tuple[int, int], jnp.ndarray]: The loaded boundaries.
        Dict[Tuple[int, int], Tuple[List[int], List[int]]]: The loaded boundary indices.
        Dict[int, jnp.ndarray]: The loaded 2D charts.
    """

    logging.info(f"Loading charts and boundaries from {charts_path}...")

    with open(charts_path + "/charts.pkl", "rb") as f:
        loaded_charts = pickle.load(f)

    with open(charts_path + "/boundaries.pkl", "rb") as f:
        loaded_boundaries = pickle.load(f)

    with open(charts_path + "/boundary_indices.pkl", "rb") as f:
        loaded_boundary_indices = pickle.load(f)
    with open(charts_path + "/charts_idxs.pkl", "rb") as f:
        loaded_charts_idxs = pickle.load(f)

    return (
        loaded_charts,
        loaded_charts_idxs,
        loaded_boundaries,
        loaded_boundary_indices,
    )


def save_charts(
    charts_path: Union[str, Path],
    charts: List[np.ndarray],
    charts_idxs: Dict[int, List[int]],
    boundaries: Dict[Tuple[int, int], jnp.ndarray],
    boundary_indices: Dict[Tuple[int, int], Tuple[List[int], List[int]]],
    chart_in_mesh_indices: Dict[int, List[int]] = None,
    mesh_in_chart_indices: Dict[int, List[int]] = None,
    verts: Dict[int, jnp.ndarray] = None,
    connectivity: Dict[int, jnp.ndarray] = None,
) -> None:
    """Save charts to a directory.

    Args:
        charts_path (Union[str, Path]): The path to the folder where the charts should be stored.
        charts (List[jnp.ndarray]): The charts to be saved.
        charts_idxs (Dict[int, List[int]]): The indices of the charts.
        boundaries (Dict[Tuple[int, int], jnp.ndarray]): The boundaries between the charts.
        boundary_indices (Dict[Tuple[int, int], Tuple[List[int], List[int]]]): The indices of the boundary points in each chart.
    Returns:
        None
    """

    Path(charts_path).mkdir(parents=True, exist_ok=True)

    with open(charts_path + "/charts.pkl", "wb") as f:
        pickle.dump(charts, f)

    with open(charts_path + "/boundaries.pkl", "wb") as f:
        pickle.dump(boundaries, f)

    with open(charts_path + "/boundary_indices.pkl", "wb") as f:
        pickle.dump(boundary_indices, f)

    with open(charts_path + "/charts_idxs.pkl", "wb") as f:
        pickle.dump(charts_idxs, f)

    if verts is not None:
        with open(charts_path + "/verts.pkl", "wb") as f:
            pickle.dump(verts, f)

    if connectivity is not None:
        with open(charts_path + "/connectivity.pkl", "wb") as f:
            pickle.dump(connectivity, f)

    if chart_in_mesh_indices is not None:
        with open(charts_path + "/chart_in_mesh_indices.pkl", "wb") as f:
            pickle.dump(chart_in_mesh_indices, f)

    if mesh_in_chart_indices is not None:
        with open(charts_path + "/mesh_in_chart_indices.pkl", "wb") as f:
            pickle.dump(mesh_in_chart_indices, f)


def get_charts(
    points: jnp.ndarray, charts_config: Dict[str, Any], verts: jnp.ndarray = None
) -> List[jnp.ndarray]:
    """
    Returns the charts from point cloud data

    Args:
        points (jnp.ndarray): The point cloud data
        charts_config (Dict[str, Any]): The configuration
    """

    if charts_config.alg == "fast_region_growing":
        charts, charts_idxs, sampled_points = fast_region_growing(
            pts=points,
            min_dist=charts_config.min_dist,
            nearest_neighbors=charts_config.nearest_neighbors,
        )

    else:
        raise NotImplementedError(f"Algorithm {charts_config.alg} not implemented")

    boundaries, boundary_indices = get_boundaries(charts)

    return charts, charts_idxs, boundaries, boundary_indices, sampled_points


def find_intersection_indices(points1, points2, tolerance=1e-8):
    """
    Find the indices of the intersection of two sets of points.

    Args:
        points1 (np.ndarray): The first set of points.
        points2 (np.ndarray): The second set of points.
        tolerance (float): Tolerance for floating-point comparisons.

    Returns:
        tuple: (intersection_indices_12, intersection_indices_21) - Indices of the intersection points
               in both arrays.
    """
    # Build KDTree for efficient nearest neighbor search
    tree2 = KDTree(points2)

    # Find matches within tolerance
    intersection_indices_12 = []
    intersection_indices_21 = []

    # For each point in points1, find all points in points2 within the tolerance radius
    for i, point in enumerate(points1):
        # Query the KDTree to find all points within tolerance
        indices = tree2.query_ball_point(point, tolerance)

        # If we found matching points
        if indices:
            for idx in indices:
                intersection_indices_21.append(i)
                intersection_indices_12.append(idx)

    # logging.info(f"Found {len(intersection_indices_12)} intersection points between arrays of shapes {points1.shape} and {points2.shape}")

    return np.array(intersection_indices_12), np.array(intersection_indices_21)


def find_verts_in_charts(
    charts: List[jnp.ndarray], verts: jnp.ndarray
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Find the indices of the vertices in the charts
    """

    mesh_in_chart_indices = {}
    chart_in_mesh_indices = {}
    for key in charts.keys():
        chart_in_mesh_indices[key], mesh_in_chart_indices[key] = (
            find_intersection_indices(charts[key], verts)
        )

    return chart_in_mesh_indices, mesh_in_chart_indices


def find_closest_points_to_mesh(mesh_pts: jnp.ndarray, pts: jnp.ndarray) -> List[int]:
    """
    Find the indices of the closest points in the mesh to the points in the charts
    """

    tree = KDTree(pts)
    distances, closest_indices = tree.query(mesh_pts, k=1)

    return distances, closest_indices
