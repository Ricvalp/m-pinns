from absl import app, logging
from ml_collections import config_flags
import wandb
import pickle
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from chart_autoencoder import (
    TrainerAutoEncoder,
    ChartsDataset,
    plot_3d_charts,
    plot_local_charts_2d,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    load_charts,
    numpy_collate,
    numpy_collate_with_distances,
)
from chart_autoencoder import (
    compute_norm_g_ginv_from_params_autoencoder,
)


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="fit/config/fit_autencoder_sphere-+.py"
)


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.profiler.log_dir).mkdir(parents=True, exist_ok=True)



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

    charts_to_fit = (
        cfg.charts_to_fit if cfg.charts_to_fit is not None else charts.keys()
    )
    
    
    chart_loaders = {
        key: DataLoader(
            dataset=ChartsDataset(charts[key], distance_matrix[key]),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=numpy_collate_with_distances,
        ) for key in charts_to_fit
    }
    
    
    recon_charts = {}
    latent_charts = {}
    for key in charts_to_fit:

        if cfg.wandb.use:
            run =wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=f"{cfg.dataset.name}_autoencoder_{key}",
            )

        trainer = TrainerAutoEncoder(
            cfg=cfg,
            chart=charts[key],
            boundary_indices={
                bkey: boundary_indices[bkey]
                for bkey in boundary_indices.keys()
                if bkey[0] == key
            },
            chart_loader=chart_loaders[key],
            chart_3d=charts[key],
            chart_key=key,
        )

        trainer.fit()

        if cfg.wandb.use:
            run.finish()

        recon_charts[key], latent_charts[key] = trainer.model_fn(charts[key])

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
        ) = compute_norm_g_ginv_from_params_autoencoder(
            model_fn=trainer.model_fn,
            decoder_fn=trainer.decoder_fn,
            chart=charts[key],
            noise_scale=0.02,
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


    with open(f"{cfg.dataset.charts_path}/charts2d.pkl", "wb") as f:
        pickle.dump(latent_charts, f)
    print(f"Saved chart points to {cfg.dataset.charts_path}/charts2d.pkl")



def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


if __name__ == "__main__":
    app.run(main)
