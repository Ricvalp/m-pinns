mkdir -p ./pinns/wave/sphere/data

PYTHONPATH=. \
    python pinns/wave/main.py \
    --config=pinns/wave/configs/sphere.py \
    --config.mode=generate_data