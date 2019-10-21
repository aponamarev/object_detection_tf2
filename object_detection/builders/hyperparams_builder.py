# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builder function to construct tf-slim arg_scope for convolution, fc ops."""
import tensorflow as tf
from object_detection.protos import hyperparams_pb2


class KerasLayerHyperparams(object):
  """
  A hyperparameter configuration object for Keras layers used in
  Object Detection models.
  """

  def __init__(self, hyperparams_config):
    """Builds keras hyperparameter config for layers based on the proto config.

    It automatically converts from Slim layer hyperparameter configs to
    Keras layer hyperparameters. Namely, it:
    - Builds Keras initializers/regularizers instead of Slim ones
    - sets weights_regularizer/initializer to kernel_regularizer/initializer
    - converts batchnorm decay to momentum
    - converts Slim l2 regularizer weights to the equivalent Keras l2 weights

    Contains a hyperparameter configuration for ops that specifies kernel
    initializer, kernel regularizer, activation. Also contains parameters for
    batch norm operators based on the configuration.

    Note that if the batch_norm parameters are not specified in the config
    (i.e. left to default) then batch norm is excluded from the config.

    Args:
      hyperparams_config: hyperparams.proto object containing
        hyperparameters.

    Raises:
      ValueError: if hyperparams_config is not of type hyperparams.Hyperparams.
    """
    if not isinstance(hyperparams_config,
                      hyperparams_pb2.Hyperparams):
      raise ValueError('hyperparams_config not of type '
                       'hyperparams_pb.Hyperparams.')

    self._batch_norm_params = None
    if hyperparams_config.HasField('batch_norm'):
      self._batch_norm_params = _build_keras_batch_norm_params(
          hyperparams_config.batch_norm)

    self._activation_fn = _build_activation_fn(hyperparams_config.activation)
    # TODO(kaftan): Unclear if these kwargs apply to separable & depthwise conv
    # (Those might use depthwise_* instead of kernel_*)
    # We should probably switch to using build_conv2d_layer and
    # build_depthwise_conv2d_layer methods instead.
    self._op_params = {
        'kernel_regularizer': _build_keras_regularizer(hyperparams_config.regularizer),
        'kernel_initializer': _build_initializer(hyperparams_config.initializer),
        'activation': _build_activation_fn(hyperparams_config.activation)
    }

  def use_batch_norm(self):
    return self._batch_norm_params is not None

  def batch_norm_params(self, **overrides):
    """Returns a dict containing batchnorm layer construction hyperparameters.

    Optionally overrides values in the batchnorm hyperparam dict. Overrides
    only apply to individual calls of this method, and do not affect
    future calls.

    Args:
      **overrides: keyword arguments to override in the hyperparams dictionary

    Returns: dict containing the layer construction keyword arguments, with
      values overridden by the `overrides` keyword arguments.
    """
    if self._batch_norm_params is None:
      new_batch_norm_params = dict()
    else:
      new_batch_norm_params = self._batch_norm_params.copy()
    new_batch_norm_params.update(overrides)
    return new_batch_norm_params

  def build_batch_norm(self, **overrides):
    """Returns a Batch Normalization layer with the appropriate hyperparams.

    If the hyperparams are configured to not use batch normalization,
    this will return a Keras Lambda layer that only applies tf.Identity,
    without doing any normalization.

    Optionally overrides values in the batch_norm hyperparam dict. Overrides
    only apply to individual calls of this method, and do not affect
    future calls.

    Args:
      training: if True, the normalization layer will normalize using the batch
       statistics. If False, the normalization layer will be frozen and will
       act as if it is being used for inference. If None, the layer
       will look up the Keras learning phase at `call` time to decide what to
       do.
      **overrides: batch normalization construction args to override from the
        batch_norm hyperparams dictionary.

    Returns: Either a FreezableBatchNorm layer (if use_batch_norm() is True),
      or a Keras Lambda layer that applies the identity (if use_batch_norm()
      is False)
    """
    if self.use_batch_norm():
      return tf.keras.layers.BatchNormalization(**self.batch_norm_params(**overrides))
    else:
      return tf.keras.layers.Lambda(tf.identity)

  def build_activation_layer(self, name='activation'):
    """Returns a Keras layer that applies the desired activation function.

    Args:
      name: The name to assign the Keras layer.
    Returns: A Keras lambda layer that applies the activation function
      specified in the hyperparam config, or applies the identity if the
      activation function is None.
    """
    if self._activation_fn:
      return tf.keras.layers.Lambda(self._activation_fn, name=name)
    else:
      return tf.keras.layers.Lambda(tf.identity, name=name)

  def params(self, include_activation=False, **overrides):
    """Returns a dict containing the layer construction hyperparameters to use.

    Optionally overrides values in the returned dict. Overrides
    only apply to individual calls of this method, and do not affect
    future calls.

    Args:
      include_activation: If False, activation in the returned dictionary will
        be set to `None`, and the activation must be applied via a separate
        layer created by `build_activation_layer`. If True, `activation` in the
        output param dictionary will be set to the activation function
        specified in the hyperparams config.
      **overrides: keyword arguments to override in the hyperparams dictionary.

    Returns: dict containing the layer construction keyword arguments, with
      values overridden by the `overrides` keyword arguments.
    """
    new_params = self._op_params.copy()
    new_params['activation'] = None
    if include_activation:
      new_params['activation'] = self._activation_fn
    if self.use_batch_norm() and self.batch_norm_params()['center']:
      new_params['use_bias'] = False
    else:
      new_params['use_bias'] = True
    new_params.update(**overrides)
    return new_params


def _build_activation_fn(activation_fn):
  """Builds a callable activation from config.

  Args:
    activation_fn: hyperparams_pb2.Hyperparams.activation

  Returns:
    Callable activation function.

  Raises:
    ValueError: On unknown activation function.
  """
  if activation_fn == hyperparams_pb2.Hyperparams.NONE:
    return None
  if activation_fn == hyperparams_pb2.Hyperparams.RELU:
    return tf.nn.relu
  if activation_fn == hyperparams_pb2.Hyperparams.RELU_6:
    return tf.nn.relu6
  raise ValueError('Unknown activation function: {}'.format(activation_fn))


def _build_keras_regularizer(regularizer):
  """Builds a keras regularizer from config.

  Args:
    regularizer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    Keras regularizer.

  Raises:
    ValueError: On unknown regularizer.
  """
  regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
  if  regularizer_oneof == 'l1_regularizer':
    return tf.keras.regularizers.l1(float(regularizer.l1_regularizer.weight))
  if regularizer_oneof == 'l2_regularizer':
    # The Keras L2 regularizer weight differs from the Slim L2 regularizer
    # weight by a factor of 2
    return tf.keras.regularizers.l2(
        float(regularizer.l2_regularizer.weight * 0.5))
  raise ValueError('Unknown regularizer function: {}'.format(regularizer_oneof))


def _build_initializer(initializer):
  """Build a tf initializer from config.

  Args:
    initializer: hyperparams_pb2.Hyperparams.regularizer proto.
    build_for_keras: Whether the initializers should be built for Keras
      operators. If false builds for Slim.

  Returns:
    tf initializer.

  Raises:
    ValueError: On unknown initializer.
  """
  initializer_oneof = initializer.WhichOneof('initializer_oneof')
  if initializer_oneof == 'truncated_normal_initializer':
    return tf.keras.initializers.TruncatedNormal(
            mean=initializer.truncated_normal_initializer.mean,
            stddev=initializer.truncated_normal_initializer.stddev
    )

  if initializer_oneof == 'random_normal_initializer':
    return tf.keras.initializers.RandomNormal(
            mean=initializer.random_normal_initializer.mean,
            stddev=initializer.random_normal_initializer.stddev
    )

  if initializer_oneof == 'variance_scaling_initializer':
    enum_descriptor = (hyperparams_pb2.VarianceScalingInitializer.
                       DESCRIPTOR.enum_types_by_name['Mode'])
    mode = enum_descriptor.values_by_number[initializer.
                                            variance_scaling_initializer.
                                            mode].name
    return tf.keras.initializers.VarianceScaling(
            scale=initializer.variance_scaling_initializer.factor,
            mode=mode.lower(),
            distribution='uniform' if initializer.variance_scaling_initializer.uniform else 'truncated_normal'
    )
  raise ValueError('Unknown initializer function: {}'.format(
      initializer_oneof))


def _build_keras_batch_norm_params(batch_norm):
  """Build a dictionary of Keras BatchNormalization params from config.

  Args:
    batch_norm: hyperparams_pb2.ConvHyperparams.batch_norm proto.

  Returns:
    A dictionary containing Keras BatchNormalization parameters.
  """
  # Note: Although decay is defined to be 1 - momentum in batch_norm,
  # decay in the slim batch_norm layers was erroneously defined and is
  # actually the same as momentum in the Keras batch_norm layers.
  # For context, see: github.com/keras-team/keras/issues/6839
  batch_norm_params = {
      'momentum': batch_norm.decay,
      'center': batch_norm.center,
      'scale': batch_norm.scale,
      'epsilon': batch_norm.epsilon,
  }
  return batch_norm_params