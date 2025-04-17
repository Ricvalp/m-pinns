import numpy as np
from sklearn.decomposition import PCA
import igl  # libigl Python bindings


def get_initial_parameterization(points_3d, method="pca"):
    """Get initial 2D parameterization for a set of 3D points.

    Args:
        points_3d: (N,3) array of 3D points
        method: Parameterization method ("pca" or "tutte")

    Returns:
        points_2d: (N,2) array of 2D parameterized points
    """
    if method == "pca":
        # Use PCA to project points onto their principal 2D plane
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points_3d)

    elif method == "tutte":
        # For this we need mesh connectivity
        # Assuming we have faces array
        uv = igl.harmonic_weights(points_3d, faces, 1)
        points_2d = uv

    return points_2d


def parameterize_charts(charts):
    """Get 2D parameterizations for all charts.

    Args:
        charts: Dict of chart_idx -> (N,3) points arrays

    Returns:
        parameterizations: Dict of chart_idx -> (N,2) parameter arrays
    """
    parameterizations = {}
    for idx, points_3d in charts.items():
        points_2d = get_initial_parameterization(points_3d)
        parameterizations[idx] = points_2d
    return parameterizations
