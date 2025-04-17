PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/sphere.py \
    --config.autoencoder_checkpoint.step=100 \
    --config.mode=train
