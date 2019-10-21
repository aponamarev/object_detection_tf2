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
r"""Constructs model, inputs, and training environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import os

import tensorflow as tf


from object_detection.utils import config_util

# A map of names to methods that help build the model.
MODEL_BUILD_UTIL_MAP = {
    'get_configs_from_pipeline_file':
        config_util.get_configs_from_pipeline_file,
    'create_pipeline_proto_from_configs':
        config_util.create_pipeline_proto_from_configs,
    'merge_external_params_with_configs':
        config_util.merge_external_params_with_configs,
    'create_train_input_fn':
        None, #inputs.create_train_input_fn,
    'create_eval_input_fn':
        None, # inputs.create_eval_input_fn,
    'create_predict_input_fn':
        None, # inputs.create_predict_input_fn,
    'detection_model_fn_base': None # model_builder.build,
}


def create_config(pipeline_config_path: str,
                  config_override=None,
                  train_steps=None,
                  sample_1_of_n_eval_examples=1,
                  override_eval_num_epochs=True,
                  use_tpu=False,
                  **kwargs):
    """

    :param pipeline_config_path: A path to a pipeline config file.
    :param config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override the config from `pipeline_config_path`.
    :param train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    :param sample_1_of_n_eval_examples: Integer representing how often an eval example
      should be sampled. If 1, will sample all examples.
    :param override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for
      eval_input.
    :param use_tpu: Boolean, whether training and evaluation should run on TPU. Only
      used if `use_tpu_estimator` is True.
    :param kwargs: Additional keyword arguments for configuration override.

    Returns:
    `configs` dictionary.
    """

    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP['merge_external_params_with_configs']

    configs = get_configs_from_pipeline_file(
            pipeline_config_path, config_override=config_override)
    kwargs.update({
        'train_steps':  train_steps,
        'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
    })
    if sample_1_of_n_eval_examples >= 1:
        kwargs.update({
            'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples
        })
    if override_eval_num_epochs:
        kwargs.update({'eval_num_epochs': 1})
        print('Forced number of epochs for all eval validations to be 1.')
    return merge_external_params_with_configs(configs, kwargs_dict=kwargs)
