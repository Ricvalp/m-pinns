PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/coil.py \
    --config.eval.checkpoint_dir="pinns/eikonal_autodecoder/coil/checkpoints/5_training_points/0lygja76" \
    --config.eval.step=9999 \
    --config.eval.use_existing_solution=True \
    --config.mode=eval