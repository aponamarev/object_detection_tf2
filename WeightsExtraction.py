# Project: WeightsExtraction.py
# Description: Designed to extract weights from an existing checkpoint and store the weights in a pickled dictionary
# Author: Alexander Ponamarev
# Created Date: 4/19/18
import os
from pickle import dump
from tensorflow.python import pywrap_tensorflow
from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, 'Path to output model directory '
                                       'where event and checkpoint files will be written.')
flags.mark_flag_as_required('model_dir')
flags.DEFINE_string('model_name', 'model.ckpt', 'Name of the file containing a frozen graph')
flags.DEFINE_string('output_path', None, 'Provide a full output path (including file name) where the weights will be stored')
flags.mark_flag_as_required('output_path')


def main(argv):

    reader = load_checkpoint(path=FLAGS.model_dir, model_name=FLAGS.model_name)
    variables = extract_variables(reader=reader)
    with open(FLAGS.output_path, "wb+") as f:
        dump(variables, f)
    print("The following variables were successfully pickled:")
    for key in variables.keys():
        print(key, "shape: ", variables[key]['shape'])

    return 0


def load_checkpoint(path: str, model_name: str) -> pywrap_tensorflow.CheckpointReader:
    # Create a path to frozen graph
    checkpoint_path = os.path.join(path, model_name)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)

    return reader


def extract_variables(reader: pywrap_tensorflow.CheckpointReader) -> dict:

    var_to_shape_map = reader.get_variable_to_shape_map()
    variables = {name: {'values': reader.get_tensor(name), 'shape': shape} for name, shape in var_to_shape_map.items()}

    return variables


if __name__ == '__main__':
    app.run(main)

