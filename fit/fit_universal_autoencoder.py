from absl import app, logging
from ml_collections import config_flags
import wandb
import pickle
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from datasets import DeformedDiskDataset
from universal_autoencoder import (
    TrainerUniversalAutoEncoder,
    numpy_collate,
    numpy_collate_with_distances,
)
from chart_autoencoder import (
    compute_norm_g_ginv_from_params_autoencoder,
)
import datasets.random_chart_dataset as chart_dataset


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="fit/config/fit_universal_autoencoder.py"
)


def main(_):
    # Load configuration
    cfg = load_cfgs(_TASK_FILE)
    
    # Create necessary directories
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.profiler.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint.checkpoint_path).mkdir(parents=True, exist_ok=True)

    # data_loader = DataLoader(
    #         dataset=DeformedDiskDataset(),
    #         batch_size=cfg.train.batch_size,
    #         shuffle=True,
    #         num_workers=8,
    #         collate_fn=numpy_collate_with_distances,
    #     )

    dataset = DeformedDiskDataset(
        seed=cfg.seed,
        num_points=cfg.dataset.num_points,
        batch_size=cfg.train.batch_size,
        disk_radius=cfg.dataset.disk_radius,
        num_control=cfg.dataset.num_control,
        deform_scale=cfg.dataset.deform_scale,
        kernel_func=getattr(chart_dataset, cfg.dataset.kernel_func),
        kernel_epsilon=cfg.dataset.kernel_epsilon,
    )

    # Initialize wandb if needed
    run = None
    if cfg.wandb.use:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"universal_autoencoder",
            config=cfg.to_dict(),  # Log the configuration
        )
        wandb_id = run.id
    else:
        wandb_id = "no_wandb_" + str(cfg.seed)

    # Create and run trainer
    trainer = TrainerUniversalAutoEncoder(
        cfg=cfg,
        dataset=dataset,
        wandb_id=wandb_id,
    )

    # Train the model
    trainer.fit()

    # If using wandb, finish the run
    if cfg.wandb.use and run is not None:
        run.finish()

    # (
    #     recon_noisy_chart,
    #     recon_chart,
    #     latent_chart,
    #     noisy_latent_chart,
    #     g,
    #     g_inv,
    #     norm_g,
    #     norm_g_inv,
    # ) = compute_norm_g_ginv_from_params_autoencoder(
    #     model_fn=trainer.model_fn,
    #     decoder_fn=trainer.decoder_fn,
    #     chart=charts[key],
    #     noise_scale=0.02,
    # )

    # plot_local_charts_2d(
    #     charts={key: noisy_latent_chart},
    #     original_charts={key: latent_chart},
    #     g={key: norm_g},
    #     boundaries_indices=boundary_indices,
    #     name=Path(cfg.figure_path)
    #     / f"charts_{key}"
    #     / f"post_{cfg.dataset.name}_noisy_latent_charts_with_g.png",
    # )

    # plot_local_charts_2d(
    #     charts={key: noisy_latent_chart},
    #     original_charts={key: latent_chart},
    #     g={key: norm_g_inv},
    #     boundaries_indices=boundary_indices,
    #     name=Path(cfg.figure_path)
    #     / f"charts_{key}"
    #     / f"post_{cfg.dataset.name}_noisy_latent_charts_with_g_inv.png",
    # )


def load_cfgs(_TASK_FILE):
    """Load configuration from file."""
    cfg = _TASK_FILE.value
    return cfg


if __name__ == "__main__":
    app.run(main)
