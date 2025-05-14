import os

import eval
import train
import generate_data
from absl import app, flags
from ml_collections import config_flags

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=false "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    "config",
    default="./configs/default.py",
    help_string="File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config)

    elif FLAGS.config.mode == "generate_data":
        generate_data.generate_data(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
