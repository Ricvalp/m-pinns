import os
from datetime import datetime
import jax


def get_last_checkpoint_dir(workdir):
    ckpts = os.listdir(workdir)
    sorted_dir_names = sorted(ckpts, key=dir_to_datetime)
    return max(sorted_dir_names)


def dir_to_datetime(dir_name):
    return datetime.strptime(dir_name, "%Y-%m-%d_%H-%M-%S")


def set_profiler(profiler_config, step, log_dir):
    # Profiling.
    if profiler_config is not None:
        if step == profiler_config.start_step:
            jax.profiler.start_trace(log_dir=log_dir)
        if step == profiler_config.end_step:
            jax.profiler.stop_trace()
