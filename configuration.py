# Author: Mikita Sazanovich

import argparse
import yaml


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('tag', type=str, help='The tag which will be included in the model output dir name.')
  parser.add_argument('--config_path', type=str, default='exps/unit/duckietown_unit.yaml')
  parser.add_argument('--output_dir_base', type=str, default='output')
  parser.add_argument('--test_checkpoint_dir', type=str, default='')  # the checkpoint dir to use for the test stage
  parser.add_argument('--skip_train', action='store_true')  # whether to skip the training stage
  parser.add_argument('--skip_test', action='store_true')  # whether to skip the test stage
  parser.add_argument('--summarize', action='store_true')  # whether to print model summaries
  parsed_args = parser.parse_args()
  return parsed_args


def load_config(path):
  with open(path, 'r') as stream:
    doc = yaml.load(stream)
    return doc['config']
