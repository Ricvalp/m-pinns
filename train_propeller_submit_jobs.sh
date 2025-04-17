#!/bin/bash

rm -rf slurm_logs_train
mkdir -p slurm_logs_train

NUM_LAYERS=(1 2)
LEARNING_RATES=(0.01 0.001 0.0001)
DECAY_STEPS=(10000)
FOURIER_EMB=(128 256 512)

job_counter=0
for num_layer in ${NUM_LAYERS[@]}; do
    for lr in ${LEARNING_RATES[@]}; do
        for decay_step in ${DECAY_STEPS[@]}; do
            for emb_dim in ${FOURIER_EMB[@]}; do
                job_counter=$((job_counter + 1))
                cat << EOF > slurm_logs_train/job_${job_counter}.slurm
#!/bin/bash
#SBATCH --job-name=train_${job_counter}
#SBATCH --output=slurm_logs_train/train_${job_counter}_%j.out
#SBATCH --error=slurm_logs_train/train_${job_counter}_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=01:00:00
#SBATCH --gpus=1

source ~/.bashrc
conda init
conda activate manifold-pinns

PYTHONPATH=. \
python pinns/eikonal_autodecoder/main.py \
--config=pinns/eikonal_autodecoder/configs/propeller.py \
--config.autoencoder_checkpoint.step=60000 \
--config.mode=train \
--config.optim.learning_rate=${lr} \
--config.arch.num_layers=${num_layer} \
--config.arch.fourier_emb.embed_dim=${emb_dim} \
--config.optim.decay_steps=${decay_step} \

EOF
                sbatch slurm_logs_train/job_${job_counter}.slurm
            done
        done
    done
done


