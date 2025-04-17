PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/sphere.py \
    --config.eval.checkpoint_dir="pinns/eikonal_autodecoder/sphere/checkpoints/best/xdplcssa" \
    --config.eval.step=29999 \
    --config.eval.use_existing_solution=False \
    --config.mode=eval