# Author: Mikita Sazanovich

import os
import unittest
import time

import configuration
import train


class TrainTest(unittest.TestCase):
  def test_dataset_iteration(self):
    config_path = os.path.join('configs', 'unit', 'duckietown_unit.yaml')
    config = configuration.load_config(config_path)
    a_train_dataset, a_test_dataset, a_test_length = train.create_image_action_dataset(config, 'domain_a')
    b_train_dataset, b_test_dataset, b_test_length = train.create_image_action_dataset(config, 'domain_b')

    for dataset in [a_train_dataset, a_test_dataset, b_train_dataset, b_test_dataset]:
      time_start = time.time()
      for batch_tuple in dataset:
        print(f'Dataset has been tested. The length of the batch tuple is {len(batch_tuple)}.')
        for batch_element in batch_tuple:
          print(batch_element.shape)
        break
      time_ended = time.time()
      time_spent = time_ended - time_start
      print(f'Time spent: {time_spent:.4f}s')

  def test_joint_train_iteration(self):
    config_path = os.path.join('configs', 'unit', 'duckietown_unit.yaml')
    config = configuration.load_config(config_path)

    trainer = train.create_models_and_trainer(config)

    a_train_dataset, a_test_dataset, a_test_length = train.create_image_action_dataset(config, 'domain_a')
    b_train_dataset, b_test_dataset, b_test_length = train.create_image_action_dataset(config, 'domain_b')

    a_dataset_iter = iter(a_train_dataset)
    b_dataset_iter = iter(b_train_dataset)
    images_a, actions_a = next(a_dataset_iter)
    images_b, _ = next(b_dataset_iter)

    time_start = time.time()
    trainer.joint_train_step(images_a, actions_a, images_b)
    time_ended = time.time()
    time_spent = time_ended - time_start
    print(f'Time spent: {time_spent:.4f}s')
