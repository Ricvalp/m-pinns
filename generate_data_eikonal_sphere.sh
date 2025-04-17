mkdir -p ./pinns/eikonal_autodecoder/sphere/data

PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/sphere.py \
    --config.mode=generate_data