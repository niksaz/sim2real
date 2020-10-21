# Author: Mikita Sazanovich

import os
import unittest
import time

import configuration
import train_unit


def print_time(start, ended):
  print(f'Time spent: {ended - start:.4f}s')


class TrainUnitTest(unittest.TestCase):
  def test_dataset_iteration(self):
    config_path = os.path.join('configs', 'unit', 'duckietown_unit.yaml')
    config = configuration.load_config(config_path)
    a_train_dataset, a_test_dataset, a_test_length = train_unit.create_image_action_dataset(config, 'domain_a')
    b_train_dataset, b_test_dataset, b_test_length = train_unit.create_image_action_dataset(config, 'domain_b')

    for dataset in [a_train_dataset, a_test_dataset, b_train_dataset, b_test_dataset]:
      time_start = time.time()
      for batch_tuple in dataset:
        print(f'Dataset has been tested. The length of the batch tuple is {len(batch_tuple)}.')
        for el in batch_tuple:
          print(el.shape)
        break
      time_ended = time.time()
      print_time(time_start, time_ended)
