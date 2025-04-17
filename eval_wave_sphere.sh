PYTHONPATH=. \
    python pinns/wave/main.py \
    --config=pinns/wave/configs/sphere.py \
    --config.eval.checkpoint_dir="pinns/wave/sphere/checkpoints/" \
    --config.eval.step=59999 \
    --config.mode=eval