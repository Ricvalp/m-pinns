import os
from pathlib import Path
import numpy as np

import imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import models
from tqdm import tqdm

from chart_autoencoder.riemann import get_metric_tensor_and_sqrt_det_g_autodecoder

from pinns.diffusion_single_gpu_autodecoder.get_dataset import get_dataset
from pinns.diffusion_single_gpu_autodecoder.utils import get_last_checkpoint_dir

from jaxpi.utils import restore_checkpoint, load_config

import jax


def evaluate(config: ml_collections.ConfigDict):

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), d_params = get_metric_tensor_and_sqrt_det_g_autodecoder(
        autoencoder_config,
        step=config.autoencoder_checkpoint.step,
        inverse=True,
    )

    x, y, u0, boundaries_x, boundaries_y, charts3d = get_dataset(
        autoencoder_config.dataset.charts_path,
        sigma=1.0,
    )

    # Initialize model
    model = models.Diffusion(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        ics=(x, y, u0),
        boundaries=(boundaries_x, boundaries_y),
    )

    # Restore the last checkpoint
    if config.eval.eval_with_last_ckpt:
        last_ckpt_dir = get_last_checkpoint_dir(config.eval.checkpoint_dir)
        ckpt_path = (Path(config.eval.checkpoint_dir) / Path(last_ckpt_dir)).resolve()
    else:
        ckpt_path = Path(config.eval.eval_checkpoint_dir).resolve()

    model.state = restore_checkpoint(model.state, ckpt_path, step=config.eval.step)
    params = model.state.params

    times = jnp.linspace(0.0, config.T, 100)
    u_preds = []

    for i in tqdm(range(len(x))):
        cur_u_preds = []
        for ti in times:
            cur_u_preds.append(
                model.u_pred_fn(jax.tree.map(lambda x: x[i], params), x[i], y[i], ti)
            )
        u_preds.append(cur_u_preds)

    d_params = [jax.tree.map(lambda x: x[i], d_params) for i in range(len(x))]

    filenames = []
    vmin = min(np.min(u_pred) for u_pred in u_preds)
    vmax = max(np.max(u_pred) for u_pred in u_preds)

    num_rows = np.ceil(np.sqrt(len(x)))
    num_cols = np.ceil(len(x) / num_rows)

    for i, u_pred in enumerate(zip(*u_preds)):
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        plt.suptitle(f"t = {times[i]:.2f}")
        for j, u in enumerate(u_pred):
            row = j // num_cols
            col = j % num_cols
            ax[row, col].scatter(
                x[j], y[j], c=u, cmap="jet", s=2.5, vmin=vmin, vmax=vmax
            )
            ax[row, col].set_xlabel("x")
            ax[row, col].set_ylabel("y")
            ax[row, col].set_title(f"Chart {j}")
        filename = config.figure_path + f"/single_gpu_diffusion_{i}.png"
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)

    # Create a GIF
    with imageio.get_writer(
        config.figure_path + "/single_gpu_diffusion.png", mode="I"
    ) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

    # 3D plots
    for angles in [(30, 45), (30, 135), (30, 225), (30, 315)]:

        filenames = []
        for i, u_pred in enumerate(zip(*u_preds)):

            fig = plt.figure(figsize=(18, 5))
            ax = fig.add_subplot(1, 1, 1, projection="3d")

            for j, u in enumerate(u_pred):
                X = decoder.apply(
                    {"params": d_params[j]}, jnp.stack([x[j], y[j]], axis=1)
                )
                ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    X[:, 2],
                    c=u,
                    cmap="jet",
                    s=2.5,
                    vmin=vmin,
                    vmax=vmax,
                )

            ax.view_init(angles[0], angles[1])

            filename = config.figure_path + f"/single_gpu_diffusion_{i}_3d.png"
            plt.savefig(filename)
            plt.close()
            filenames.append(filename)

        # Create a GIF
        with imageio.get_writer(
            config.figure_path + f"/single_gpu_diffusion_3d_{angles[1]}.png", mode="I"
        ) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    for filename in filenames:
        os.remove(filename)
