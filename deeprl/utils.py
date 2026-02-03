"""Common functions you may find useful in your implementation."""

import tensorflow as tf


def get_soft_target_model_updates(target, source, tau):
    r"""Perform soft update of target network weights.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have the same architecture as source model.
    source: keras.models.Model
      The source model. Should have the same architecture as target model.
    tau: float
      The weight of the source weights to the target weights used
      during update.
    """
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign((1. - tau) * target_var + tau * source_var)


def get_hard_target_model_updates(target, source):
    """Perform hard update of target network weights.

    The source weights are copied directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have the same architecture as source model.
    source: keras.models.Model
      The source model. Should have the same architecture as target model.
    """
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign(source_var)


def check_tensorflow_version():
    """Check that TensorFlow version is compatible (>=2.0.0).
    
    Returns
    -------
    bool
        True if TensorFlow version is >= 2.0.0, raises RuntimeError otherwise.
    """
    tf_version = tf.__version__
    major_version = int(tf_version.split('.')[0])
    if major_version < 2:
        raise RuntimeError(f"TensorFlow version {tf_version} is not supported. Please use TensorFlow >= 2.0.0.")
    return True
