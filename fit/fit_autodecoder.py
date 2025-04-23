from absl import app, logging
from ml_collections import config_flags
import wandb
from pathlib import Path
import numpy as np
from chart_autoencoder import TrainerAutoDecoder
from chart_autoencoder import (
    plot_3d_charts,
    plot_local_charts_2d,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    load_charts,
)
from chart_autoencoder import (
    compute_norm_g_ginv_from_params,
)


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="fit/config/fit_autodecoder_coil.py"
)


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.profiler.log_dir).mkdir(parents=True, exist_ok=True)

    # if cfg.wandb.use:
    #     wandb.init(
    #         project=cfg.wandb.project,
    #         entity=cfg.wandb.entity,
    #     )

    charts, charts_idxs, boundaries, boundary_indices, charts2d = load_charts(
        charts_path=cfg.dataset.charts_path,
    )

    logging.info(f"Loaded charts. Got {len(charts)} charts")

    plot_html_3d_charts(
        charts=charts,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.html",
    )
    plot_html_3d_boundaries(
        boundaries=boundaries,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_boundaries.html",
    )
    plot_3d_charts(
        charts=charts,
        gt_charts=None,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.png",
    )

    distance_matrix = np.load(
        cfg.dataset.distance_matrix_path, allow_pickle=True
    ).item()
    logging.info("Loaded distance matrix")

    umap_charts = np.load(cfg.umap.umap_embeddings_path, allow_pickle=True).item()
    logging.info("Loaded UMAP embeddings")

    plot_local_charts_2d(
        charts=umap_charts,
        boundaries_indices=boundary_indices,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_umap_charts.png",
    )

    charts_to_fit = (
        cfg.charts_to_fit if cfg.charts_to_fit is not None else charts.keys()
    )
    recon_charts = {}
    latent_charts = {}
    for key in charts_to_fit:

        if cfg.wandb.use:
            run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=f"{cfg.dataset.name}_autoencoder_{key}",
            )

        trainer = TrainerAutoDecoder(
            cfg=cfg,
            chart=umap_charts[key],
            boundary_indices={
                bkey: boundary_indices[bkey]
                for bkey in boundary_indices.keys()
                if bkey[0] == key
            },
            chart_3d=charts[key],
            distances_matrix=distance_matrix[key],
            chart_key=key,
        )

        if cfg.train.warmup_steps > 0:
            trainer.warmup(num_steps=cfg.train.warmup_steps)
        trainer.fit()

        recon_charts[key] = trainer.decoder_fn()
        latent_charts[key] = trainer.state.params["points"]

        plot_3d_charts(
            charts={key: recon_charts[key]},
            gt_charts={key: charts[key]},
            name=Path(cfg.figure_path) / f"charts_{key}" / "post_chart_3d.png",
        )
        (
            recon_noisy_chart,
            recon_chart,
            latent_chart,
            noisy_latent_chart,
            g,
            g_inv,
            norm_g,
            norm_g_inv,
        ) = compute_norm_g_ginv_from_params(
            params=trainer.state.params,
            decoder_fn=trainer.decoder_apply_fn,
        )

        plot_local_charts_2d(
            charts={key: noisy_latent_chart},
            original_charts={key: latent_chart},
            g={key: norm_g},
            boundaries_indices=boundary_indices,
            name=Path(cfg.figure_path)
            / f"charts_{key}"
            / f"post_{cfg.dataset.name}_noisy_latent_charts_with_g.png",
        )

        plot_local_charts_2d(
            charts={key: noisy_latent_chart},
            original_charts={key: latent_chart},
            g={key: norm_g_inv},
            boundaries_indices=boundary_indices,
            name=Path(cfg.figure_path)
            / f"charts_{key}"
            / f"post_{cfg.dataset.name}_noisy_latent_charts_with_g_inv.png",
        )

        del trainer

        if cfg.wandb.use:
            run.finish()

    plot_html_3d_charts(
        charts=recon_charts,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_charts_3d.html",
    )
    plot_3d_charts(
        charts=recon_charts,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_charts_3d.png",
    )
    plot_local_charts_2d(
        charts=latent_charts,
        boundaries_indices=boundary_indices,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_latent_charts.png",
    )


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


if __name__ == "__main__":
    app.run(main)
