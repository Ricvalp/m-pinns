mkdir -p ./pinns/eikonal_autodecoder/propeller/data

PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/propeller.py \
    --config.mode=generate_data