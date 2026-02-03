"""Loss functions.

Note: The DQNAgent class uses tf.keras.losses.Huber() by default.
These custom implementations are provided as alternatives that allow
more control over the delta/max_grad parameter.

Usage example:
    from deeprl.objectives import mean_huber_loss
    
    # Use as a custom loss function
    agent.loss_func = mean_huber_loss
"""

import tensorflow as tf


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate the Huber loss using TensorFlow (element-wise).
    
    The Huber loss is less sensitive to outliers than MSE.
    For small errors (< delta), it behaves like MSE.
    For large errors (>= delta), it behaves like MAE.
    
    Parameters
    ----------
    y_true: tf.Tensor
      The ground truth (target) values.
    y_pred: tf.Tensor
      The predicted values.
    delta: float, optional
      Threshold at which the loss changes from quadratic to linear.
      Default is 1.0, matching tf.keras.losses.Huber default.

    Returns
    -------
    loss: tf.Tensor
      The computed Huber loss (element-wise, not reduced).
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = 0.5 * tf.square(tf.minimum(abs_error, delta))
    linear = delta * (abs_error - tf.minimum(abs_error, delta))
    return quadratic + linear


def mean_huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate the mean Huber loss using TensorFlow.
    
    This is equivalent to tf.keras.losses.Huber(delta=delta, reduction='auto').
    
    Parameters
    ----------
    y_true: tf.Tensor
      The ground truth (target) values.
    y_pred: tf.Tensor
      The predicted values.
    delta: float, optional
      Threshold at which the loss changes from quadratic to linear.
      Default is 1.0, matching tf.keras.losses.Huber default.
    
    Returns
    -------
    mean_loss: tf.Tensor
      The mean Huber loss (scalar).
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, delta))

