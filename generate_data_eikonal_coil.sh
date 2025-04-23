#!/bin/bash

# Default value for N if not provided
N=${1:-4}

PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/coil.py \
    --config.mode=generate_data \
    --config.N=$N \
    --config.training.batches_path="pinns/eikonal_autodecoder/coil/data/"
