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
"""Binary to run train and evaluation on object detection model."""

from absl import app
from absl import flags
from absl import logging
import os, sys
import matplotlib
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from PIL import Image
import numpy as np

from object_detection import model_lib_v2 as model_lib
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

if sys.platform == 'darwin':
    matplotlib.use('MacOSX')

logging.set_verbosity(logging.DEBUG)

flags.DEFINE_string(
        'img_path', None, 'Path to the image file '
                           'to be processed by the neural network.')
flags.DEFINE_string(
        'model_dir', None, 'Path to output model directory '
                           'where event and checkpoint files will be written.')
flags.DEFINE_string(
        'path_to_labels', None, 'Path to labels '
                           'to be used to name objects detected by the network.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                                                  'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_string(
        'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                                '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                                'writing resulting metrics to `model_dir`.')
FLAGS = flags.FLAGS

IMAGE_SIZE = (12, 8)


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    if not os.path.exists(FLAGS.model_dir):
        raise FileExistsError("The following forlder doesn't exist: {}".format(
                FLAGS.model_dir
        ))
    flags.mark_flag_as_required('img_path')
    if not os.path.exists(FLAGS.img_path):
        raise FileExistsError("The following file doesn't exist: {}".format(
                FLAGS.img_path
        ))
    flags.mark_flag_as_required('path_to_labels')
    if not os.path.exists(FLAGS.path_to_labels):
        raise FileExistsError("The following file doesn't exist: {}".format(
                FLAGS.path_to_labels
        ))
    flags.mark_flag_as_required('pipeline_config_path')
    if not os.path.exists(FLAGS.pipeline_config_path):
        raise FileExistsError("The following file doesn't exist: {}".format(
                FLAGS.pipeline_config_path
        ))

    category_index = label_map_util.create_category_index_from_labelmap(FLAGS.path_to_labels, use_display_name=True)

    image = Image.open(FLAGS.img_path)
    image_np = load_image_into_numpy_array(image)
    test_data = np.expand_dims(image_np, axis=0).astype(np.float32)

    reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.model_dir+"/model.ckpt")

    config = model_lib.create_config(
            pipeline_config_path=FLAGS.pipeline_config_path,
    )
    (model, loss) = model_builder.build(
            model_config=config['model'],
            is_training=True,
            add_summaries=True)

    model.build(tf.TensorShape((None, None, None, 3)))

    model.feature_extractor.restore_from_tf_checkpoint_fn(reader)
    model.box_predictor.restore_from_tf_checkpoint_fn(reader)

    test_outputs = model(test_data)
    detection_dict = test_outputs['detections']

    output_dict = {
        'num_detections': int(detection_dict['num_detections'][0]),
        'detection_classes': detection_dict['detection_classes'].numpy().astype(np.int64)[0],
        'detection_boxes': detection_dict['detection_boxes'].numpy()[0],
        'detection_scores': detection_dict['detection_scores'].numpy()[0]
    }

    print("\nObjects detected:", output_dict['num_detections'])
    _ = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'] + 1,
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=.35)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)


    print("Success!")

    return 0


def load_image_into_numpy_array(image) -> np.ndarray:
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)




if __name__ == '__main__':
    app.run(main)
