mkdir -p ./pinns/eikonal_universal_autoencoder/coil/data

PYTHONPATH=. \
    python pinns/eikonal_universal_autoencoder/main.py \
    --config=pinns/eikonal_universal_autoencoder/configs/coil.py \
    --config.autoencoder_checkpoint.step=250000 \
    --config.mode=generate_data