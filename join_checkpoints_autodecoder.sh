#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME=$1

PYTHONPATH=. \
    python ./fit/checkpoints/join_checkpoints_autodecoders.py "$DATASET_NAME"