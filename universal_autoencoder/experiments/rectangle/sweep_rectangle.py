import os
from itertools import product
from pathlib import Path


sweep_params = {
    "config.dataset.num_supernodes": [32, 64, 128, 256, 64, 64, 64, 64, 64, 128, 512, 32, 64, 128, 256, 64, 64],
    "config.encoder_supernodes_cfg.max_degree": [5, 5, 5, 5, 5, 5, 5, 10, 10, 15, 5, 5, 5, 5, 5, 5, 5],
    "config.encoder_supernodes_cfg.gnn_dim": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 64, 64, 128, 128, 64],
    "config.encoder_supernodes_cfg.perc_dim": [256, 256, 256, 256, 512, 512, 256, 256, 256, 256, 512, 512, 128, 64, 64, 128, 128],
    "config.encoder_supernodes_cfg.perc_num_heads": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    "config.modulated_siren_cfg.omega_0": [1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 3.0, 3.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0],
    "config.modulated_siren_cfg.num_layers": [3, 4, 4, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4],
    "config.modulated_siren_cfg.hidden_dim": [256, 128, 256, 256, 256, 256, 256, 256, 256, 256, 32, 256, 128, 256, 256, 256, 256],
    "config.encoder_supernodes_cfg.enc_depth": [4, 4, 4, 4, 4, 4, 2, 2, 8, 8, 8, 4, 4, 4, 4, 4, 4],
    "config.train.batch_size": [64, 64, 64, 64, 64, 128, 128, 128, 64, 64, 64, 128, 128, 128, 128, 128, 32, 32, 32, 64, 64, 64],
    "config.train.lr": [1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
    "config.train.optimizer": ["cosine_decay", "cosine_decay", "cosine_decay", "cosine_decay", "cosine_decay", "adam", "adam", "adam", "adam", "adam", "adam", "adam", "adam", "adam", "adam"],
}

# sweep_params_2 = {
#     "config.encoder_supernodes_cfg.coord_enc_dim": [32, 64, 128, 64],
#     "config.modulated_siren_cfg.omega_0": [1.0, 1.0, 3.0, 5.0],
#     "config.modulated_siren_cfg.num_layers": [2, 2, 3, 4],
# }

# Generate all combinations of parameters

# all_combinations = list(zip(*sweep_params.values()))
all_combinations = list(zip(*sweep_params.values())) # + list(zip(*sweep_params_2.values())) + list(zip(*sweep_params_3.values()))

print(all_combinations)

dryrun = False  # Set to True to test the script without running jobs
job_name = "sweep_rectangle"

# Paths
script_path = "$HOME/mpinns/universal_autoencoder/experiments/rectangle/fit_universal_autoencoder_rectangle.py"
output_dir = f"/scratch-shared/rvalperga/mpinns/universal_autoencoder/experiments/rectangle/{job_name}"
template_path = "/home/rvalperga/mpinns/universal_autoencoder/experiments/rectangle/template.slurm"

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Read the template
with open(template_path, "r") as template_file:
    template = template_file.read()
    
num_jobs = len(all_combinations)
SBU_COSTS = {
    "gpu_a100": 128,
    "gpu_h100": 192,
}
partition = "gpu_a100"
time_limit = "05:00:00"
# Calculate the total cost
if partition not in SBU_COSTS:
    raise ValueError(f"Unknown partition: {partition}. Available partitions are: {list(SBU_COSTS.keys())}")
# convert time_limit into hours
time_limit_hours = int(time_limit.split(":")[0]) + int(time_limit.split(":")[1]) / 60


total_cost = num_jobs * SBU_COSTS[partition] * time_limit_hours

# Generate job scripts
for i, param_combination in enumerate(all_combinations):
    default_script_args = f""
    for j, param_name in enumerate(sweep_params.keys()):
        param_value = param_combination[j]
        default_script_args += f" --{param_name}={param_value}"
    # Create the job name
    print(default_script_args)
    cur_job_name = f"{job_name}_{i}"
    
    job_script = template.format(
        script_path=script_path,
        job_name=cur_job_name,
        partition=partition,
        time_limit=time_limit,
        output_path=output_dir,
        script_args=default_script_args
    )

    # Write the job script to a file
    job_file_path = os.path.join(output_dir, f"{cur_job_name}.sh")
    with open(job_file_path, "w") as job_file:
        job_file.write(job_script)
        
    # If dryrun is True only store the files, otherwise also submit the jobs
    if not dryrun:
        os.system(f"sbatch {job_file_path}")
    else:
        print(f"Dry run: {job_file_path} would be submitted.")
        
    print(f"Job {job_name} {'submitted' if not dryrun else 'not submitted'}.")

print(f"Generated job scripts in {output_dir}")

print(f"Total cost for {num_jobs} jobs: {total_cost} SBU")