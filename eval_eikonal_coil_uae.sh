PYTHONPATH=. \
    python pinns/eikonal_universal_autoencoder/main.py \
    --config=pinns/eikonal_universal_autoencoder/configs/coil.py \
    --config.eval.checkpoint_dir="pinns/eikonal_universal_autoencoder/coil/checkpoints/vx4g8wic" \
    --config.eval.step=90000 \
    --config.eval.use_existing_solution=False \
    --config.mode=eval