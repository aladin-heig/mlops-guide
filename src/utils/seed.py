import os

import keras
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set all random seeds and enable deterministic operations"""
    os.environ["PYTHONHASHSEED"] = str(seed)

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
