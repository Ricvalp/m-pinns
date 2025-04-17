PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/propeller.py \
    --config.eval.checkpoint_dir="pinns/eikonal_autodecoder/propeller/checkpoints/best/vgbnnq2u" \
    --config.eval.step=89999 \
    --config.eval.use_existing_solution=True \
    --config.mode=eval