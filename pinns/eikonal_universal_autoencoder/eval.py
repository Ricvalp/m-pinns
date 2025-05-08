from pathlib import Path
import logging

import jax.numpy as jnp
import numpy as np
import ml_collections
import models
from tqdm import tqdm

from chart_autoencoder import (
    get_metric_tensor_and_sqrt_det_g_autodecoder,
    load_charts,
    find_intersection_indices,
    find_closest_points_to_mesh,
    get_metric_tensor_and_sqrt_det_g_universal_autodecoder,
    load_charts3d,
)

from pinns.eikonal_universal_autoencoder.get_dataset import get_dataset, get_eikonal_gt_solution
from pinns.eikonal_universal_autoencoder.utils import get_last_checkpoint_dir

from jaxpi.utils import restore_checkpoint, load_config
from jaxpi.solution import get_final_solution, load_solution, save_solution

from plot import (
    plot_3d_level_curves,
    plot_3d_solution,
    plot_charts_solution,
    plot_correlation,
)

import jax


def evaluate(config: ml_collections.ConfigDict):

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)
    Path(config.eval.solution_path).mkdir(parents=True, exist_ok=True)

    charts_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )
    model_config = load_config(
        Path(config.eval.checkpoint_dir) / "cfg.json",
    )

    eval_config = config.eval

    (
        loaded_charts3d,
        loaded_charts_idxs,
        loaded_boundaries,
        loaded_boundary_indices,
    ) = load_charts3d(config.dataset.charts_path)

    charts_mu = np.zeros((len(loaded_charts3d.keys()), 3))
    charts_std = np.zeros((len(loaded_charts3d.keys()), ))
    for key in loaded_charts3d.keys():
        mu = loaded_charts3d[key].mean(axis=0)
        std = loaded_charts3d[key].std()
        charts_mu[key] = mu
        charts_std[key] = std
        loaded_charts3d[key] = (loaded_charts3d[key] - mu) / std

    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), (conditionings, d_params) = get_metric_tensor_and_sqrt_det_g_universal_autodecoder(
        autoencoder_cfg=charts_config,
        cfg=config,
        charts=loaded_charts3d,
        inverse=True,
    )

    x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, charts3d = get_dataset(
        charts_path=config.dataset.charts_path,
        N=config.N,
        idxs=config.idxs,
    )

    num_charts = len(x)

    model = models.Eikonal(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        conditionings=conditionings,
        bcs_charts=jnp.array(list(bcs.keys())),
        boundaries=(boundaries_x, boundaries_y),
        num_charts=num_charts,
        mu=charts_mu,
        std=charts_std,
    )

    if eval_config.eval_with_last_ckpt:
        last_ckpt_dir = get_last_checkpoint_dir(eval_config.checkpoint_dir)
        ckpt_path = (Path(eval_config.checkpoint_dir) / Path(last_ckpt_dir)).resolve()
    else:
        ckpt_path = Path(eval_config.checkpoint_dir).resolve()

    charts, charts_idxs, boundaries, boundary_indices, charts2d = load_charts(
        charts_path=config.dataset.charts_path,
    )

    eval_name = eval_config.checkpoint_dir.split("/")[-1]

    if eval_config.use_existing_solution:
        logging.info("WARNING: Loading existing solution")
        pts, sol, u_preds, mesh_sol, gt_sol = load_solution(
            eval_config.solution_path + f"/eikonal_solution_{eval_name}.npy"
        )

    else:

        model.state = restore_checkpoint(model.state, ckpt_path, step=eval_config.step)
        params = model.state.params

        u_preds = {}

        u_pred_fn = jax.jit(model.u_pred_fn)

        logging.info("Evaluating the solution on the charts")
        for i in tqdm(range(len(x))):
            u_preds[i] = u_pred_fn(
                jax.tree.map(lambda x: x[i], params), x[i], y[i]
            )  # model.

        logging.info("Joining solutions")
        pts, sol = get_final_solution(
            charts=charts,
            charts_idxs=charts_idxs,
            u_preds=u_preds,
        )

        mesh_pts, gt_sol = get_eikonal_gt_solution(
            charts_path=config.dataset.charts_path,
        )

        # gt_sol_pts_idxs, _ = find_intersection_indices(
        #     mesh_pts,
        #     pts,
        # )

        _, gt_sol_pts_idxs = find_closest_points_to_mesh(
            mesh_pts,
            pts,
        )

        assert len(gt_sol_pts_idxs) == len(
            mesh_pts
        ), "The number of points in the mesh and the number of intersection points don't match. Probably due to numerical errors."

        mesh_sol = sol[gt_sol_pts_idxs]

        save_solution(
            eval_config.solution_path + f"/eikonal_solution_{eval_name}.npy",
            pts,
            sol,
            u_preds,
            mesh_sol,
            gt_sol,
        )

    if eval_config.plot_everything:
        plot_charts_solution(x, y, u_preds, name=config.figure_path + "/eikonal.png")

        for angles in [(30, 45)]:  # , (30, 135), (30, 225), (30, 315)]:
            plot_3d_solution(
                pts,
                sol,
                angles,
                config.figure_path + f"/eikonal_3d_{angles[1]}.png",
                s=2.5,
            )

        # for tol in [1e-2, 5e-2, 1e-1, 5e-1]:
        #     plot_3d_level_curves(
        #         pts,
        #         sol,
        #         tol,
        #         name=config.figure_path + f"/eikonal_3d_level_curves_{tol}.png",
        #     )

        for angles in [(30, 45)]:  # , (30, 135), (30, 225), (30, 315)]:
            plot_3d_solution(
                mesh_pts,
                gt_sol,
                angles,
                config.figure_path + f"/gt_eikonal_3d_{angles[1]}.png",
                s=15,
            )
            plot_3d_solution(
                mesh_pts,
                gt_sol - sol[gt_sol_pts_idxs],
                angles,
                config.figure_path + f"/difference_eikonal_3d_{angles[1]}.png",
                s=15,
            )

    MSE = jnp.mean((mesh_sol - gt_sol) ** 2)
    print(f"MSE: {MSE}")
    print(f"Correlation: {jnp.corrcoef(mesh_sol, gt_sol)[0, 1]}")

    known_solution = np.load(
        config.dataset.charts_path + "/known_solution.npy", allow_pickle=True
    ).item()
    known_solution = np.concatenate(
        [known_solution[key] for key in known_solution.keys()]
    )

    plot_correlation(
        mesh_sol,
        gt_sol,
        known_solution,
        name=config.figure_path + "/eikonal_correlation",
    )
