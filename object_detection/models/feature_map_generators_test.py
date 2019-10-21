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

"""Tests for feature map generators."""

from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.models import feature_map_generators
from object_detection.protos import hyperparams_pb2

INCEPTION_V2_LAYOUT = {
    'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 256],
    'anchor_strides': [16, 32, 64, -1, -1, -1],
    'layer_target_norm': [20.0, -1, -1, -1, -1, -1],
}

INCEPTION_V3_LAYOUT = {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128],
    'anchor_strides': [16, 32, 64, -1, -1, -1],
    'aspect_ratios': [1.0, 2.0, 1.0/2, 3.0, 1.0/3]
}

EMBEDDED_SSD_MOBILENET_V1_LAYOUT = {
    'from_layer': ['Conv2d_11_pointwise', 'Conv2d_13_pointwise', '', '', ''],
    'layer_depth': [-1, -1, 512, 256, 256],
    'conv_kernel_size': [-1, -1, 3, 3, 2],
}

SSD_MOBILENET_V1_WEIGHT_SHARED_LAYOUT = {
    'from_layer': ['Conv2d_13_pointwise', '', '', ''],
    'layer_depth': [-1, 256, 256, 256],
}


class MultiResolutionFeatureMapGeneratorTest(tf.test.TestCase):

    def _build_conv_hyperparams(self):
        conv_hyperparams = hyperparams_pb2.Hyperparams()
        conv_hyperparams_text_proto = """
            regularizer {
                l2_regularizer {
                }
            }
            initializer {
                truncated_normal_initializer {
                }
            }
        """
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
        return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

    def _build_feature_map_generator(self, feature_map_layout):
        return feature_map_generators.KerasMultiResolutionFeatureMaps(
                feature_map_layout=feature_map_layout,
                depth_multiplier=1,
                min_depth=32,
                insert_1x1_conv=True,
                freeze_batchnorm=False,
                is_training=True,
                conv_hyperparams=self._build_conv_hyperparams(),
                name='FeatureMaps')

    def test_get_expected_feature_map_shapes_with_inception_v2(self):
        image_features = {
            'Mixed_3c': tf.random.uniform([4, 28, 28, 256], dtype=tf.float32),
            'Mixed_4c': tf.random.uniform([4, 14, 14, 576], dtype=tf.float32),
            'Mixed_5c': tf.random.uniform([4, 7, 7, 1024], dtype=tf.float32)
        }
        feature_map_generator = self._build_feature_map_generator(
                feature_map_layout=INCEPTION_V2_LAYOUT
        )
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'Mixed_3c': (4, 28, 28, 256),
            'Mixed_4c': (4, 14, 14, 576),
            'Mixed_5c': (4, 7, 7, 1024),
            'Mixed_5c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
            'Mixed_5c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
            'Mixed_5c_2_Conv2d_5_3x3_s2_256': (4, 1, 1, 256)
        }

        out_feature_map_shapes = dict(
                (key, value.shape) for key, value in feature_maps.items()
        )
        self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

    def test_get_expected_feature_map_shapes_with_inception_v2_use_depthwise(self):
        image_features = {
            'Mixed_3c': tf.random.uniform([4, 28, 28, 256], dtype=tf.float32),
            'Mixed_4c': tf.random.uniform([4, 14, 14, 576], dtype=tf.float32),
            'Mixed_5c': tf.random.uniform([4, 7, 7, 1024], dtype=tf.float32)
        }
        layout_copy = INCEPTION_V2_LAYOUT.copy()
        layout_copy['use_depthwise'] = True
        feature_map_generator = self._build_feature_map_generator(
                feature_map_layout=layout_copy
        )
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'Mixed_3c': (4, 28, 28, 256),
            'Mixed_4c': (4, 14, 14, 576),
            'Mixed_5c': (4, 7, 7, 1024),
            'Mixed_5c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
            'Mixed_5c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
            'Mixed_5c_2_Conv2d_5_3x3_s2_256': (4, 1, 1, 256)
        }

        out_feature_map_shapes = dict(
                (key, value.shape) for key, value in feature_maps.items()
        )
        self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

    def test_get_expected_feature_map_shapes_use_explicit_padding(self):
        image_features = {
            'Mixed_3c': tf.random.uniform([4, 28, 28, 256], dtype=tf.float32),
            'Mixed_4c': tf.random.uniform([4, 14, 14, 576], dtype=tf.float32),
            'Mixed_5c': tf.random.uniform([4, 7, 7, 1024], dtype=tf.float32)
        }
        layout_copy = INCEPTION_V2_LAYOUT.copy()
        layout_copy['use_explicit_padding'] = True
        feature_map_generator = self._build_feature_map_generator(
            feature_map_layout=layout_copy
        )
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'Mixed_3c': (4, 28, 28, 256),
            'Mixed_4c': (4, 14, 14, 576),
            'Mixed_5c': (4, 7, 7, 1024),
            'Mixed_5c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
            'Mixed_5c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
            'Mixed_5c_2_Conv2d_5_3x3_s2_256': (4, 1, 1, 256)
        }

        out_feature_map_shapes = dict(
                (key, value.shape) for key, value in feature_maps.items())
        self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

    def test_get_expected_feature_map_shapes_with_inception_v3(self):
        image_features = {
            'Mixed_5d': tf.random.uniform([4, 35, 35, 256], dtype=tf.float32),
            'Mixed_6e': tf.random.uniform([4, 17, 17, 576], dtype=tf.float32),
            'Mixed_7c': tf.random.uniform([4, 8, 8, 1024], dtype=tf.float32)
        }

        feature_map_generator = self._build_feature_map_generator(
            feature_map_layout=INCEPTION_V3_LAYOUT
        )
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'Mixed_5d': (4, 35, 35, 256),
            'Mixed_6e': (4, 17, 17, 576),
            'Mixed_7c': (4, 8, 8, 1024),
            'Mixed_7c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
            'Mixed_7c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
            'Mixed_7c_2_Conv2d_5_3x3_s2_128': (4, 1, 1, 128)
        }

        out_feature_map_shapes = dict(
                (key, value.shape) for key, value in feature_maps.items())
        self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

    def test_get_expected_feature_map_shapes_with_embedded_ssd_mobilenet_v1(self):
        image_features = {
            'Conv2d_11_pointwise': tf.random.uniform([4, 16, 16, 512], dtype=tf.float32),
            'Conv2d_13_pointwise': tf.random.uniform([4, 8, 8, 1024], dtype=tf.float32),
        }

        feature_map_generator = self._build_feature_map_generator(
            feature_map_layout=EMBEDDED_SSD_MOBILENET_V1_LAYOUT,
        )
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'Conv2d_11_pointwise': (4, 16, 16, 512),
            'Conv2d_13_pointwise': (4, 8, 8, 1024),
            'Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512': (4, 4, 4, 512),
            'Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256': (4, 2, 2, 256),
            'Conv2d_13_pointwise_2_Conv2d_4_2x2_s2_256': (4, 1, 1, 256)
        }

        out_feature_map_shapes = dict(
                (key, value.shape) for key, value in feature_maps.items()
        )
        self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

    def test_get_expected_variable_names_with_inception_v2(self):
        image_features = {
            'Mixed_3c': tf.random.uniform([4, 28, 28, 256], dtype=tf.float32),
            'Mixed_4c': tf.random.uniform([4, 14, 14, 576], dtype=tf.float32),
            'Mixed_5c': tf.random.uniform([4, 7, 7, 1024], dtype=tf.float32)
        }
        feature_map_generator = self._build_feature_map_generator(
            feature_map_layout=INCEPTION_V2_LAYOUT
        )
        feature_maps = feature_map_generator(image_features)

        expected_keras_variables = {'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/kernel',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/bias',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/kernel',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/bias',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/kernel',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/bias',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/kernel',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/bias',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/kernel',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/bias',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/kernel',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/bias'}

        actual_variable_set = {var.name.split(":")[0] for var in feature_map_generator.variables}

        self.assertSetEqual(expected_keras_variables, actual_variable_set)

    def test_get_expected_variable_names_with_inception_v2_use_depthwise(self):
        image_features = {
            'Mixed_3c': tf.random.uniform([4, 28, 28, 256], dtype=tf.float32),
            'Mixed_4c': tf.random.uniform([4, 14, 14, 576], dtype=tf.float32),
            'Mixed_5c': tf.random.uniform([4, 7, 7, 1024], dtype=tf.float32)
        }
        layout_copy = INCEPTION_V2_LAYOUT.copy()
        layout_copy['use_depthwise'] = True
        feature_map_generator = self._build_feature_map_generator(
            feature_map_layout=layout_copy
        )
        feature_maps = feature_map_generator(image_features)

        actual_variable_set = {w.name.split(":")[0] for w in feature_map_generator.weights}

        expected_keras_variables = {'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/kernel',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/bias',
                                    ('FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_depthwise_conv/'
                                     'depthwise_kernel'), ('FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_depthwise_conv/'
                                                           'bias'),
                                    'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/kernel',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/bias',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/kernel',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/bias',
                                    ('FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_depthwise_conv/'
                                     'depthwise_kernel'), ('FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_depthwise_conv/'
                                                           'bias'),
                                    'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/kernel',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/bias',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/kernel',
                                    'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/bias',
                                    ('FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_depthwise_conv/'
                                     'depthwise_kernel'), ('FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_depthwise_conv/'
                                                           'bias'),
                                    'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/kernel',
                                    'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/bias'}

        self.assertSetEqual(expected_keras_variables, actual_variable_set)


@parameterized.parameters({'use_native_resize_op': True},
                          {'use_native_resize_op': False})
class FPNFeatureMapGeneratorTest(tf.test.TestCase, parameterized.TestCase):

    def _build_conv_hyperparams(self):
        conv_hyperparams = hyperparams_pb2.Hyperparams()
        conv_hyperparams_text_proto = """
        regularizer {
            l2_regularizer {
            }
        }
        initializer {
            truncated_normal_initializer {
            }
        }
        """
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
        return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

    def _build_feature_map_generator(
            self, image_features, depth,
            use_bounded_activations=False,
            use_native_resize_op=False,
            use_explicit_padding=False,
            use_depthwise=False):
        return feature_map_generators.KerasFpnTopDownFeatureMaps(
                num_levels=len(image_features),
                depth=depth,
                is_training=True,
                conv_hyperparams=self._build_conv_hyperparams(),
                freeze_batchnorm=False,
                use_depthwise=use_depthwise,
                use_explicit_padding=use_explicit_padding,
                use_bounded_activations=use_bounded_activations,
                use_native_resize_op=use_native_resize_op,
                scope=None,
                name='FeatureMaps',
        )

    def test_get_expected_feature_map_shapes(self, use_native_resize_op):
        image_features = [
            ('block2', tf.random.uniform([4, 8, 8, 256], dtype=tf.float32)),
            ('block3', tf.random.uniform([4, 4, 4, 256], dtype=tf.float32)),
            ('block4', tf.random.uniform([4, 2, 2, 256], dtype=tf.float32)),
            ('block5', tf.random.uniform([4, 1, 1, 256], dtype=tf.float32))
        ]
        feature_map_generator = self._build_feature_map_generator(
            image_features=image_features,
            depth=128,
            use_native_resize_op=use_native_resize_op)
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'top_down_block2': (4, 8, 8, 128),
            'top_down_block3': (4, 4, 4, 128),
            'top_down_block4': (4, 2, 2, 128),
            'top_down_block5': (4, 1, 1, 128)
        }

        out_feature_map_shapes = {key: value.shape
                                  for key, value in feature_maps.items()}
        self.assertDictEqual(out_feature_map_shapes, expected_feature_map_shapes)

    def test_get_expected_feature_map_shapes_with_explicit_padding(self, use_native_resize_op):
        image_features = [
            ('block2', tf.random.uniform([4, 8, 8, 256], dtype=tf.float32)),
            ('block3', tf.random.uniform([4, 4, 4, 256], dtype=tf.float32)),
            ('block4', tf.random.uniform([4, 2, 2, 256], dtype=tf.float32)),
            ('block5', tf.random.uniform([4, 1, 1, 256], dtype=tf.float32))
        ]
        feature_map_generator = self._build_feature_map_generator(
            image_features=image_features,
            depth=128,
            use_explicit_padding=True,
            use_native_resize_op=use_native_resize_op)
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'top_down_block2': (4, 8, 8, 128),
            'top_down_block3': (4, 4, 4, 128),
            'top_down_block4': (4, 2, 2, 128),
            'top_down_block5': (4, 1, 1, 128)
        }

        out_feature_map_shapes = {key: value.shape
                                  for key, value in feature_maps.items()}
        self.assertDictEqual(out_feature_map_shapes, expected_feature_map_shapes)

    # def test_use_bounded_activations_add_operations(self, use_native_resize_op):
    #
    #     image_features = [('block2', tf.random.uniform([4, 8, 8, 256], dtype=tf.float32)),
    #                         ('block3', tf.random.uniform([4, 4, 4, 256], dtype=tf.float32)),
    #                         ('block4', tf.random.uniform([4, 2, 2, 256], dtype=tf.float32)),
    #                         ('block5', tf.random.uniform([4, 1, 1, 256], dtype=tf.float32))]
    #
    #     feature_map_generator = self._build_feature_map_generator(
    #             image_features=image_features,
    #             depth=128,
    #             use_bounded_activations=True,
    #             use_native_resize_op=use_native_resize_op)
    #
    #     feature_maps = feature_map_generator(image_features)
    #
    #     expected_added_operations = dict.fromkeys([
    #         'FeatureMaps/top_down/clip_by_value/clip_by_value',
    #         'FeatureMaps/top_down/clip_by_value_1/clip_by_value',
    #         'FeatureMaps/top_down/clip_by_value_2/clip_by_value',
    #         'FeatureMaps/top_down/clip_by_value_3/clip_by_value',
    #         'FeatureMaps/top_down/clip_by_value_4/clip_by_value',
    #         'FeatureMaps/top_down/clip_by_value_5/clip_by_value',
    #         'FeatureMaps/top_down/clip_by_value_6/clip_by_value',
    #     ])
    #
    #     op_names = {op.name: None for op in feature_map_generator.layers}
    #     self.assertDictContainsSubset(expected_added_operations, op_names)

    def test_get_expected_feature_map_shapes_with_depthwise(self, use_native_resize_op):
        image_features = [
            ('block2', tf.random.uniform([4, 8, 8, 256], dtype=tf.float32)),
            ('block3', tf.random.uniform([4, 4, 4, 256], dtype=tf.float32)),
            ('block4', tf.random.uniform([4, 2, 2, 256], dtype=tf.float32)),
            ('block5', tf.random.uniform([4, 1, 1, 256], dtype=tf.float32))
        ]
        feature_map_generator = self._build_feature_map_generator(
                image_features=image_features,
                depth=128,
                use_depthwise=True,
                use_native_resize_op=use_native_resize_op
        )
        feature_maps = feature_map_generator(image_features)

        expected_feature_map_shapes = {
            'top_down_block2': (4, 8, 8, 128),
            'top_down_block3': (4, 4, 4, 128),
            'top_down_block4': (4, 2, 2, 128),
            'top_down_block5': (4, 1, 1, 128)
        }

        out_feature_map_shapes = {key: value.shape
                                  for key, value in feature_maps.items()}
        self.assertDictEqual(out_feature_map_shapes, expected_feature_map_shapes)

    def test_get_expected_variable_names(self, use_native_resize_op):
        image_features = [
            ('block2', tf.random.uniform([4, 8, 8, 256], dtype=tf.float32)),
            ('block3', tf.random.uniform([4, 4, 4, 256], dtype=tf.float32)),
            ('block4', tf.random.uniform([4, 2, 2, 256], dtype=tf.float32)),
            ('block5', tf.random.uniform([4, 1, 1, 256], dtype=tf.float32))
        ]
        feature_map_generator = self._build_feature_map_generator(
            image_features=image_features,
            depth=128,
            use_native_resize_op=use_native_resize_op
        )
        feature_maps = feature_map_generator(image_features)

        expected_keras_variables = {
            'FeatureMaps/top_down/projection_1/kernel',
            'FeatureMaps/top_down/projection_1/bias',
            'FeatureMaps/top_down/projection_2/kernel',
            'FeatureMaps/top_down/projection_2/bias',
            'FeatureMaps/top_down/projection_3/kernel',
            'FeatureMaps/top_down/projection_3/bias',
            'FeatureMaps/top_down/projection_4/kernel',
            'FeatureMaps/top_down/projection_4/bias',
            'FeatureMaps/top_down/smoothing_1_conv/kernel',
            'FeatureMaps/top_down/smoothing_1_conv/bias',
            'FeatureMaps/top_down/smoothing_2_conv/kernel',
            'FeatureMaps/top_down/smoothing_2_conv/bias',
            'FeatureMaps/top_down/smoothing_3_conv/kernel',
            'FeatureMaps/top_down/smoothing_3_conv/bias'
        }

        actual_variable_set = {var.name.split(":")[0] for var in feature_map_generator.variables}
        self.assertSetEqual(expected_keras_variables, actual_variable_set)

    def test_get_expected_variable_names_with_depthwise(self, use_native_resize_op):
        image_features = [
            ('block2', tf.random.uniform([4, 8, 8, 256], dtype=tf.float32)),
            ('block3', tf.random.uniform([4, 4, 4, 256], dtype=tf.float32)),
            ('block4', tf.random.uniform([4, 2, 2, 256], dtype=tf.float32)),
            ('block5', tf.random.uniform([4, 1, 1, 256], dtype=tf.float32))
        ]
        feature_map_generator = self._build_feature_map_generator(
            image_features=image_features,
            depth=128,
            use_depthwise=True,
            use_native_resize_op=use_native_resize_op
        )
        feature_maps = feature_map_generator(image_features)

        expected_keras_variables = {
            'FeatureMaps/top_down/projection_1/kernel',
            'FeatureMaps/top_down/projection_1/bias',
            'FeatureMaps/top_down/projection_2/kernel',
            'FeatureMaps/top_down/projection_2/bias',
            'FeatureMaps/top_down/projection_3/kernel',
            'FeatureMaps/top_down/projection_3/bias',
            'FeatureMaps/top_down/projection_4/kernel',
            'FeatureMaps/top_down/projection_4/bias',
            'FeatureMaps/top_down/smoothing_1_depthwise_conv/depthwise_kernel',
            'FeatureMaps/top_down/smoothing_1_depthwise_conv/pointwise_kernel',
            'FeatureMaps/top_down/smoothing_1_depthwise_conv/bias',
            'FeatureMaps/top_down/smoothing_2_depthwise_conv/depthwise_kernel',
            'FeatureMaps/top_down/smoothing_2_depthwise_conv/pointwise_kernel',
            'FeatureMaps/top_down/smoothing_2_depthwise_conv/bias',
            'FeatureMaps/top_down/smoothing_3_depthwise_conv/depthwise_kernel',
            'FeatureMaps/top_down/smoothing_3_depthwise_conv/pointwise_kernel',
            'FeatureMaps/top_down/smoothing_3_depthwise_conv/bias'
        }

        actual_variable_set = {var.name.split(":")[0] for var in feature_map_generator.variables}
        self.assertSetEqual(expected_keras_variables, actual_variable_set)


class GetDepthFunctionTest(tf.test.TestCase):

    def test_return_min_depth_when_multiplier_is_small(self):

        depth_fn = feature_map_generators.get_depth_fn(depth_multiplier=0.5,
                                                       min_depth=16)
        self.assertEqual(depth_fn(16), 16)

    def test_return_correct_depth_with_multiplier(self):

        depth_fn = feature_map_generators.get_depth_fn(depth_multiplier=0.5,
                                                       min_depth=16)
        self.assertEqual(depth_fn(64), 32)


if __name__ == '__main__':
    tf.test.main()
