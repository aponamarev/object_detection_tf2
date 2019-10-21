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

"""Functions to generate a list of feature maps based on image features.

Provides several feature map generators that can be used to build object
detection feature extractors.

Object detection feature extractors usually are built by stacking two components
- A base feature extractor such as Inception V3 and a feature map generator.
Feature map generators build on the base feature extractors and produce a list
of final feature maps.
"""
import collections
import functools
import tensorflow as tf
from object_detection.utils import ops

# Activation bound used for TPU v1. Activations will be clipped to
# [-ACTIVATION_BOUND, ACTIVATION_BOUND] when training with
# use_bounded_activations enabled.
ACTIVATION_BOUND = 6.0


def get_depth_fn(depth_multiplier, min_depth):
  """Builds a callable to compute depth (output channels) of conv filters.

  Args:
    depth_multiplier: a multiplier for the nominal depth.
    min_depth: a lower bound on the depth of filters.

  Returns:
    A callable that takes in a nominal depth and returns the depth to use.
  """
  def multiply_depth(depth):
    new_depth = int(depth * depth_multiplier)
    return max(new_depth, min_depth)

  return multiply_depth


def create_conv_block(
    use_depthwise, kernel_size, padding, stride, layer_name, conv_hyperparams,
    is_training, freeze_batchnorm, depth):
  """Create Keras layers for depthwise & non-depthwise convolutions.

  Args:
    use_depthwise: Whether to use depthwise separable conv instead of regular
      conv.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      filters. Can be an int if both values are the same.
    padding: One of 'VALID' or 'SAME'.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      convolution stride. Can be an int if both strides are the same.
    layer_name: String. The name of the layer.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    is_training: Indicates whether the feature generator is in training mode.
    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    depth: Depth of output feature maps.

  Returns:
    A list of conv layers.
  """
  layers = []
  if use_depthwise:
    kwargs = conv_hyperparams.params()
    # Both the regularizer and initializer apply to the depthwise layer,
    # so we remap the kernel_* to depthwise_* here.
    kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
    kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
    layers.append(
        tf.keras.layers.SeparableConv2D(
            depth, [kernel_size, kernel_size],
            depth_multiplier=1,
            padding=padding,
            strides=stride,
            name=layer_name + '_depthwise_conv',
            **kwargs))
  else:
    layers.append(tf.keras.layers.Conv2D(
        depth,
        [kernel_size, kernel_size],
        padding=padding,
        strides=stride,
        name=layer_name + '_conv',
        **conv_hyperparams.params()))
  layers.append(
      conv_hyperparams.build_batch_norm(
          training=(is_training and not freeze_batchnorm),
          name=layer_name + '_batchnorm'))
  layers.append(
      conv_hyperparams.build_activation_layer(
          name=layer_name))
  return layers


class KerasMultiResolutionFeatureMaps(tf.keras.Model):
  """Generates multi resolution feature maps from input image features.

  A Keras model that generates multi-scale feature maps for detection as in the
  SSD papers by Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.

  More specifically, when called on inputs it performs the following two tasks:
  1) If a layer name is provided in the configuration, returns that layer as a
     feature map.
  2) If a layer name is left as an empty string, constructs a new feature map
     based on the spatial shape and depth configuration. Note that the current
     implementation only supports generating new layers using convolution of
     stride 2 resulting in a spatial resolution reduction by a factor of 2.
     By default convolution kernel size is set to 3, and it can be customized
     by caller.

  An example of the configuration for Inception V3:
  {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128]
  }

  When this feature generator object is called on input image_features:
    Args:
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  """

  def __init__(self,
               feature_map_layout,
               depth_multiplier,
               min_depth,
               insert_1x1_conv,
               is_training,
               conv_hyperparams,
               freeze_batchnorm,
               name=None):
    """Constructor.

    Args:
      feature_map_layout: Dictionary of specifications for the feature map
        layouts in the following format (Inception V2/V3 respectively):
        {
          'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
          'layer_depth': [-1, -1, -1, 512, 256, 128]
        }
        or
        {
          'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
          'layer_depth': [-1, -1, -1, 512, 256, 128]
        }
        If 'from_layer' is specified, the specified feature map is directly used
        as a box predictor layer, and the layer_depth is directly infered from
        the feature map (instead of using the provided 'layer_depth' parameter).
        In this case, our convention is to set 'layer_depth' to -1 for clarity.
        Otherwise, if 'from_layer' is an empty string, then the box predictor
        layer will be built from the previous layer using convolution
        operations. Note that the current implementation only supports
        generating new layers using convolutions of stride 2 (resulting in a
        spatial resolution reduction by a factor of 2), and will be extended to
        a more flexible design. Convolution kernel size is set to 3 by default,
        and can be customized by 'conv_kernel_size' parameter (similarily,
        'conv_kernel_size' should be set to -1 if 'from_layer' is specified).
        The created convolution operation will be a normal 2D convolution by
        default, and a depthwise convolution followed by 1x1 convolution if
        'use_depthwise' is set to True.
      depth_multiplier: Depth multiplier for convolutional layers.
      min_depth: Minimum depth for convolutional layers.
      insert_1x1_conv: A boolean indicating whether an additional 1x1
        convolution should be inserted before shrinking the feature map.
      is_training: Indicates whether the feature generator is in training mode.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(KerasMultiResolutionFeatureMaps, self).__init__(name=name)

    self.feature_map_layout = feature_map_layout
    self.convolutions = []

    depth_fn = get_depth_fn(depth_multiplier, min_depth)

    base_from_layer = ''
    use_explicit_padding = False
    if 'use_explicit_padding' in feature_map_layout:
      use_explicit_padding = feature_map_layout['use_explicit_padding']
    use_depthwise = False
    if 'use_depthwise' in feature_map_layout:
      use_depthwise = feature_map_layout['use_depthwise']
    for index, from_layer in enumerate(feature_map_layout['from_layer']):
      net = []
      layer_depth = feature_map_layout['layer_depth'][index]
      conv_kernel_size = 3
      if 'conv_kernel_size' in feature_map_layout:
        conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
      if from_layer:
        base_from_layer = from_layer
      else:
        if insert_1x1_conv:
          layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(
              base_from_layer, index, depth_fn(layer_depth / 2))
          net.append(tf.keras.layers.Conv2D(depth_fn(layer_depth / 2),
                                            [1, 1],
                                            padding='SAME',
                                            strides=1,
                                            name=layer_name + '_conv',
                                            **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  # training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name))

        layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(
            base_from_layer, index, conv_kernel_size, conv_kernel_size,
            depth_fn(layer_depth))
        stride = 2
        padding = 'SAME'
        if use_explicit_padding:
          padding = 'VALID'
          # We define this function here while capturing the value of
          # conv_kernel_size, to avoid holding a reference to the loop variable
          # conv_kernel_size inside of a lambda function
          def fixed_padding(features, kernel_size=conv_kernel_size):
            return ops.fixed_padding(features, kernel_size)
          net.append(tf.keras.layers.Lambda(fixed_padding))
        # TODO(rathodv): Add some utilities to simplify the creation of
        # Depthwise & non-depthwise convolutions w/ normalization & activations
        if use_depthwise:
          net.append(tf.keras.layers.DepthwiseConv2D(
              [conv_kernel_size, conv_kernel_size],
              depth_multiplier=1,
              padding=padding,
              strides=stride,
              name=layer_name + '_depthwise_conv',
              **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  # training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_depthwise_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name + '_depthwise'))

          net.append(tf.keras.layers.Conv2D(depth_fn(layer_depth), [1, 1],
                                            padding='SAME',
                                            strides=1,
                                            name=layer_name + '_conv',
                                            **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  # training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name))

        else:
          net.append(tf.keras.layers.Conv2D(
              depth_fn(layer_depth),
              [conv_kernel_size, conv_kernel_size],
              padding=padding,
              strides=stride,
              name=layer_name + '_conv',
              **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  # training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name))

      # Until certain bugs are fixed in checkpointable lists,
      # this net must be appended only once it's been filled with layers
      self.convolutions.append(net)

  def call(self, image_features):
    """Generate the multi-resolution feature maps.

    Executed when calling the `.__call__` method on input.

    Args:
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
    """
    feature_maps = []
    feature_map_keys = []

    for index, from_layer in enumerate(self.feature_map_layout['from_layer']):
      if from_layer:
        feature_map = image_features[from_layer]
        feature_map_keys.append(from_layer)
      else:
        feature_map = feature_maps[-1]
        for layer in self.convolutions[index]:
          feature_map = layer(feature_map)
        layer_name = self.convolutions[index][-1].name
        feature_map_keys.append(layer_name)
      feature_maps.append(feature_map)
    return collections.OrderedDict(
        [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])


class KerasFpnTopDownFeatureMaps(tf.keras.Model):
  """Generates Keras based `top-down` feature maps for Feature Pyramid Networks.

  See https://arxiv.org/abs/1612.03144 for details.
  """

  def __init__(self,
               num_levels,
               depth,
               is_training,
               conv_hyperparams,
               freeze_batchnorm,
               use_depthwise=False,
               use_explicit_padding=False,
               use_bounded_activations=False,
               use_native_resize_op=False,
               scope=None,
               name=None):
    """Constructor.

    Args:
      num_levels: the number of image features.
      depth: depth of output feature maps.
      is_training: Indicates whether the feature generator is in training mode.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      use_depthwise: whether to use depthwise separable conv instead of regular
        conv.
      use_explicit_padding: whether to use explicit padding.
      use_bounded_activations: Whether or not to clip activations to range
        [-ACTIVATION_BOUND, ACTIVATION_BOUND]. Bounded activations better lend
        themselves to quantized inference.
      use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op
        for the upsampling process instead of reshape and broadcasting
        implementation.
      scope: A scope name to wrap this op under.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(KerasFpnTopDownFeatureMaps, self).__init__(name=name)

    self.scope = scope if scope else 'top_down'
    self.top_layers = []
    self.residual_blocks = []
    self.top_down_blocks = []
    self.reshape_blocks = []
    self.conv_layers = []

    padding = 'VALID' if use_explicit_padding else 'SAME'
    stride = 1
    kernel_size = 3
    def clip_by_value(features):
      return tf.clip_by_value(features, -ACTIVATION_BOUND, ACTIVATION_BOUND)

    # top layers
    self.top_layers.append(tf.keras.layers.Conv2D(
        depth, [1, 1], strides=stride, padding=padding,
        name='projection_%d' % num_levels,
        **conv_hyperparams.params(use_bias=True)))
    if use_bounded_activations:
      self.top_layers.append(tf.keras.layers.Lambda(
          clip_by_value, name='clip_by_value'))

    for level in reversed(range(num_levels - 1)):
      # to generate residual from image features
      residual_net = []
      # to preprocess top_down (the image feature map from last layer)
      top_down_net = []
      # to reshape top_down according to residual if necessary
      reshaped_residual = []
      # to apply convolution layers to feature map
      conv_net = []
      # residual block
      residual_net.append(tf.keras.layers.Conv2D(
          depth, [1, 1], padding=padding, strides=1,
          name='projection_%d' % (level + 1),
          **conv_hyperparams.params(use_bias=True)))
      if use_bounded_activations:
        residual_net.append(tf.keras.layers.Lambda(
            clip_by_value, name='clip_by_value'))

      # top-down block
      # TODO (b/128922690): clean-up of ops.nearest_neighbor_upsampling
      if use_native_resize_op:
        def resize_nearest_neighbor(image):
          image_shape = image.shape.as_list()
          return tf.image.resize(
              image, [image_shape[1] * 2, image_shape[2] * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        top_down_net.append(tf.keras.layers.Lambda(
            resize_nearest_neighbor, name='nearest_neighbor_upsampling'))
      else:
        def nearest_neighbor_upsampling(image):
          return ops.nearest_neighbor_upsampling(image, scale=2)
        top_down_net.append(tf.keras.layers.Lambda(
            nearest_neighbor_upsampling, name='nearest_neighbor_upsampling'))

      # reshape block
      if use_explicit_padding:
        def reshape(inputs):
          residual_shape = tf.shape(input=inputs[0])
          return inputs[1][:, :residual_shape[1], :residual_shape[2], :]
        reshaped_residual.append(
            tf.keras.layers.Lambda(reshape, name='reshape'))

      # down layers
      if use_bounded_activations:
        conv_net.append(tf.keras.layers.Lambda(
            clip_by_value, name='clip_by_value'))

      if use_explicit_padding:
        def fixed_padding(features, kernel_size=kernel_size):
          return ops.fixed_padding(features, kernel_size)
        conv_net.append(tf.keras.layers.Lambda(
            fixed_padding, name='fixed_padding'))

      layer_name = 'smoothing_%d' % (level + 1)
      conv_block = create_conv_block(
          use_depthwise, kernel_size, padding, stride, layer_name,
          conv_hyperparams, is_training, freeze_batchnorm, depth)
      conv_net.extend(conv_block)

      self.residual_blocks.append(residual_net)
      self.top_down_blocks.append(top_down_net)
      self.reshape_blocks.append(reshaped_residual)
      self.conv_layers.append(conv_net)

  def call(self, image_features):
    """Generate the multi-resolution feature maps.

    Executed when calling the `.__call__` method on input.

    Args:
      image_features: list of tuples of (tensor_name, image_feature_tensor).
        Spatial resolutions of succesive tensors must reduce exactly by a factor
        of 2.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
    """
    output_feature_maps_list = []
    output_feature_map_keys = []
    with tf.name_scope(self.scope):
      top_down = image_features[-1][1]
      for layer in self.top_layers:
        top_down = layer(top_down)
      output_feature_maps_list.append(top_down)
      output_feature_map_keys.append('top_down_%s' % image_features[-1][0])

      num_levels = len(image_features)
      for index, level in enumerate(reversed(range(num_levels - 1))):
        residual = image_features[level][1]
        top_down = output_feature_maps_list[-1]
        for layer in self.residual_blocks[index]:
          residual = layer(residual)
        for layer in self.top_down_blocks[index]:
          top_down = layer(top_down)
        for layer in self.reshape_blocks[index]:
          top_down = layer([residual, top_down])
        top_down += residual
        for layer in self.conv_layers[index]:
          top_down = layer(top_down)
        output_feature_maps_list.append(top_down)
        output_feature_map_keys.append('top_down_%s' % image_features[level][0])
    return collections.OrderedDict(reversed(
        list(zip(output_feature_map_keys, output_feature_maps_list))))
