mkdir -p ./pinns/eikonal_universal_autoencoder/bunny/data

PYTHONPATH=. \
    python pinns/eikonal_universal_autoencoder/main.py \
    --config=pinns/eikonal_universal_autoencoder/configs/bunny.py \
    --config.autoencoder_checkpoint.step=550000 \
    --config.mode=generate_data