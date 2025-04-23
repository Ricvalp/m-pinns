#!/bin/bash

# Get N_VALUES from command line arguments, or use default if not provided
if [ $# -eq 0 ]; then
    N_VALUES=(12 13)  # Default values
else
    # Convert space-separated string to array
    IFS=' ' read -r -a N_VALUES <<< "$1"
fi

# Array of seeds for multiple runs
SEEDS=(0 1 2)

for N in "${N_VALUES[@]}"; do
    echo "--------------------------------"
    echo "--------------------------------"
    echo "Running ablation with N = $N"
    
    # # Step 1: Generate data with the current N value (only once per N)
    # echo "Generating data with N = $N..."
    # ./generate_data_eikonal_coil.sh "$N"
    
    # Step 2: Train model with the same N value but different seeds
    for SEED in "${SEEDS[@]}"; do
        echo " ------------------------- seed = $SEED"
        PYTHONPATH=. \
            python pinns/eikonal_autodecoder/main.py \
            --config=pinns/eikonal_autodecoder/configs/coil.py \
            --config.autoencoder_checkpoint.step=60000 \
            --config.mode=train \
            --config.N="$N" \
            --config.seed="$SEED" \
            --config.plot=False \
            --config.wandb.project="PINN-Eikonal-Coil-Ablation" \
            --config.saving.checkpoint_dir="pinns/eikonal_autodecoder/coil/checkpoints/ablation_N_${N}_seed_${SEED}" \
            --config.wandb.name="ablation_N_${N}_seed_${SEED}" \
        
        echo "Completed training for N = $N, seed = $SEED"
        echo "--------------------------------"
    done
done

echo "Ablation study complete!"