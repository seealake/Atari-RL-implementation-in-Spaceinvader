"""Common functions you may find useful in your implementation."""

import semver
import tensorflow as tf

def get_uninitialized_variables(variables=None):
    """Return a list of uninitialized tf variables for TensorFlow 2.x.

    Parameters
    ----------
    variables: tf.Variable, list(tf.Variable), optional
      Filter variable list to only those that are uninitialized. If no
      variables are specified the list of all variables in the graph
      will be used.

    Returns
    -------
    list(tf.Variable)
      List of uninitialized tf variables.
    """
    if variables is None:
        try:
            variables = tf.compat.v1.global_variables()
        except AttributeError:
            print("Warning: tf.compat.v1.global_variables() is not available. Using tf.global_variables() instead.")
            variables = tf.global_variables()

    uninitialized = []
    for var in variables:
        try:
            tf.assert_variables_initialized([var])
        except tf.errors.FailedPreconditionError:
            uninitialized.append(var)
        except Exception as e:
            print(f"Warning: Unexpected error when checking initialization of variable {var.name}: {e}")

    return uninitialized


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
    """Check that TensorFlow version is compatible (>=2.0.0)."""
    tf_version = tf.__version__
    if not semver.match(tf_version, ">=2.0.0"):
        raise RuntimeError(f"TensorFlow version {tf_version} is not supported. Please use TensorFlow >= 2.0.0.")
