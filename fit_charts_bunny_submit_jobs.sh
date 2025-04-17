#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <total_charts> <num_jobs>"
    exit 1
fi

TOTAL_CHARTS=$1
NUM_JOBS=$2

rm -rf slurm_logs_fit_charts_bunny
mkdir -p slurm_logs_fit_charts_bunny

CHARTS_PER_JOB=$((TOTAL_CHARTS / NUM_JOBS))
REMAINDER=$((TOTAL_CHARTS % NUM_JOBS))

current_datetime=$(date +"%Y%m%d_%H%M%S")

job_counter=0
start_chart=0

for ((i=0; i<NUM_JOBS; i++)); do
    end_chart=$((start_chart + CHARTS_PER_JOB - 1))
    if [ $i -lt $REMAINDER ]; then
        end_chart=$((end_chart + 1))
    fi

    chart_range=$(seq -s, $start_chart $end_chart)
    job_counter=$((job_counter + 1))

    cat << EOF > slurm_logs_fit_charts_bunny/job_${job_counter}.slurm
#!/bin/bash
#SBATCH --job-name=fit_charts_bunny_${job_counter}
#SBATCH --output=slurm_logs_fit_charts_bunny/bunny_${job_counter}_%j.out
#SBATCH --error=slurm_logs_fit_charts_bunny/bunny_${job_counter}_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --gpus=1

source ~/.bashrc
conda init
conda activate manifold-pinns

PYTHONPATH=. \
python fit/fit_autodecoder.py \
--config=fit/config/fit_autodecoder_bunny.py \
--config.charts_to_fit='(${chart_range})' \
--config.figure_path=./figures/${current_datetime}/bunny${start_chart}-${end_chart}
EOF

    sbatch slurm_logs_fit_charts_bunny/job_${job_counter}.slurm
    start_chart=$((end_chart + 1))
done