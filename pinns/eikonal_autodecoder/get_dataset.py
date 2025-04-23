import numpy as np
from jax import random
import igl
import jax.numpy as jnp
import logging

from chart_autoencoder.get_charts import load_charts, find_verts_in_charts


def get_dataset(charts_path, N=100, idxs=None):

    (
        loaded_charts3d,
        loaded_charts_idxs,
        loaded_boundaries,
        loaded_boundary_indices,
        loaded_charts2d,
    ) = load_charts(charts_path)

    x = {}
    y = {}

    for chart_key in loaded_charts2d.keys():
        x[chart_key] = loaded_charts2d[chart_key][:, 0]
        y[chart_key] = loaded_charts2d[chart_key][:, 1]

    boundaries_x = {}
    boundaries_y = {}

    for key in loaded_boundary_indices.keys():
        start_boundary_indices = np.array(loaded_boundary_indices[key])

        starting_chart = key[0]
        starting_chart_points = loaded_charts2d[starting_chart][start_boundary_indices]
        ending_chart = key[1]

        if starting_chart not in boundaries_x:
            boundaries_x[starting_chart] = {}
        if starting_chart not in boundaries_y:
            boundaries_y[starting_chart] = {}

        boundaries_x[starting_chart][ending_chart] = starting_chart_points[:, 0]
        boundaries_y[starting_chart][ending_chart] = starting_chart_points[:, 1]

    bcs_x, bcs_y, bcs = get_eikonal_bcs(
        charts_path=charts_path, x=x, y=y, charts3d=loaded_charts3d, N=N, idxs=idxs
    )

    return x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, loaded_charts3d


def get_eikonal_bcs(charts_path, x, y, charts3d, N=50, seed=37, idxs=None):

    verts = np.load(charts_path + "/verts.pkl", allow_pickle=True)
    connectivity = np.load(charts_path + "/connectivity.pkl", allow_pickle=True)

    if N > verts.shape[0]:
        N = verts.shape[0]

    Y_eg = igl.exact_geodesic(
        verts, connectivity, np.array([0]), np.arange(verts.shape[0])
    )
    n_nodes = verts.shape[0]

    if idxs is None:
        key = random.PRNGKey(seed)
        idx_train = random.choice(key, n_nodes, (N,), replace=False)
    else:
        idx_train = jnp.array(idxs[-N:])

    idx_train = jnp.sort(idx_train)
    Y = Y_eg[idx_train]
    bcs_points = verts[idx_train]

    bcs_x = {}
    bcs_y = {}
    bcs = {}

    chart_in_mesh_indices, mesh_in_chart_indices = find_verts_in_charts(
        charts3d, bcs_points
    )

    logging.info("gt solution indices: %s", idx_train)

    for chart_key in mesh_in_chart_indices.keys():
        if (
            len(mesh_in_chart_indices[chart_key]) > 0
            and len(chart_in_mesh_indices[chart_key]) > 0
        ):
            bcs_x[chart_key] = x[chart_key][mesh_in_chart_indices[chart_key]]
            bcs_y[chart_key] = y[chart_key][mesh_in_chart_indices[chart_key]]
            bcs[chart_key] = Y[chart_in_mesh_indices[chart_key]]

    return bcs_x, bcs_y, bcs


def get_eikonal_gt_solution(charts_path):

    verts = np.load(charts_path + "/verts.pkl", allow_pickle=True)
    connectivity = np.load(charts_path + "/connectivity.pkl", allow_pickle=True)

    Y_eg = igl.exact_geodesic(
        verts, connectivity, np.array([0]), np.arange(verts.shape[0])
    )

    return verts, Y_eg
