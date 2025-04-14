"""Loss functions."""

#import tensorflow as tf
#import semver

import tensorflow as tf
import semver

def huber_loss(y_true, y_pred, max_grad=1.0):
    """
    Calculate the Huber loss using TensorFlow.
    
    Parameters
    ----------
    y_true: tf.Tensor
      The ground truth (target) values.
    y_pred: tf.Tensor
      The predicted values.
    max_grad: float, optional
      Threshold at which the loss changes from quadratic to linear.

    Returns
    -------
    loss: tf.Tensor
      The computed Huber loss.
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic_part = tf.minimum(abs_error, max_grad) ** 2 * 0.5
    linear_part = max_grad * (abs_error - tf.minimum(abs_error, max_grad))
    loss = quadratic_part + linear_part
    return loss


def mean_huber_loss(y_true, y_pred, max_grad=1.0):
    """
    Calculate the mean Huber loss using TensorFlow.
    
    Parameters
    ----------
    y_true: tf.Tensor
      The ground truth (target) values.
    y_pred: tf.Tensor
      The predicted values.
    max_grad: float, optional
      Threshold at which the loss changes from quadratic to linear.
    
    Returns
    -------
    mean_loss: tf.Tensor
      The mean Huber loss.
    """
    loss = huber_loss(y_true, y_pred, max_grad)
    return tf.reduce_mean(loss)

