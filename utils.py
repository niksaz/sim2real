# Author: Mikita Sazanovich

import time
import os

import tensorflow as tf
import numpy as np


def create_output_dirs(output_dir_base, model, tag, dirs_to_create):
  output_dir_name = f'{tag}-{time.strftime("%Y%m%d%H%M%S")}'
  output_dir = os.path.join(output_dir_base, model, output_dir_name)
  os.makedirs(output_dir, exist_ok=False)
  dir_paths = []
  for dir_to_create in dirs_to_create:
    dir_path = os.path.join(output_dir, dir_to_create)
    os.makedirs(dir_path, exist_ok=False)
    dir_paths.append(dir_path)
  return dir_paths


def fix_random_seeds(seed):
  tf.random.set_seed(seed)
  np.random.seed(seed)
