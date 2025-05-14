PYTHONPATH=. \
    python pinns/eikonal_universal_autoencoder/main.py \
    --config=pinns/eikonal_universal_autoencoder/configs/bunny.py \
    --config.eval.checkpoint_dir="pinns/eikonal_universal_autoencoder/bunny/checkpoints/mbpy0hur/" \
    --config.eval.step=90000 \
    --config.eval.use_existing_solution=False \
    --config.mode=eval