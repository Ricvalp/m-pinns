PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/sphere.py \
    --config.eval.checkpoint_dir="pinns/eikonal_autodecoder/sphere/checkpoints/dq3i2970" \
    --config.eval.step=65000 \
    --config.eval.use_existing_solution=False \
    --config.mode=eval