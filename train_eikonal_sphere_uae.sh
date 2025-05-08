PYTHONPATH=. \
    python pinns/eikonal_universal_autoencoder/main.py \
    --config=pinns/eikonal_universal_autoencoder/configs/sphere.py \
    --config.autoencoder_checkpoint.step=300000 \
    --config.mode=train
