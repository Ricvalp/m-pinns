PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/coil.py \
    --config.eval.checkpoint_dir="pinns/eikonal_autodecoder/coil/checkpoints/paspfqsd" \
    --config.eval.step=14999 \
    --config.eval.use_existing_solution=False \
    --config.mode=eval