PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/bunny.py \
    --config.autoencoder_checkpoint.step=60000 \
    --config.mode=train 
