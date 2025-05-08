mkdir -p ./pinns/eikonal_universal_autoencoder/sphere/data

PYTHONPATH=. \
    python pinns/eikonal_universal_autoencoder/main.py \
    --config=pinns/eikonal_universal_autoencoder/configs/sphere.py \
    --config.mode=generate_data