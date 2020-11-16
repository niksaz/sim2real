# Author: Mikita Sazanovich

import logging
import os
import pickle
import time

import numpy as np
import tensorflow as tf


def setup_logging():
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s', datefmt='%Y-%m-%d %I-%M-%S')
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  logger.addHandler(stream_handler)


def create_output_dirs(output_dir_base, model, tag, dirs_to_create):
  model_dir = os.path.join(output_dir_base, model)
  os.makedirs(model_dir, exist_ok=True)
  output_dir_name = f'{tag}-{time.strftime("%Y%m%d%H%M%S")}'
  output_dir = os.path.join(model_dir, output_dir_name)
  os.makedirs(output_dir, exist_ok=False)
  dir_paths = []
  for dir_to_create in dirs_to_create:
    dir_path = os.path.join(output_dir, dir_to_create)
    os.makedirs(dir_path, exist_ok=False)
    dir_paths.append(dir_path)
  return output_dir, dir_paths


def load_pickle_fin(pickle_path):
  with open(pickle_path, 'rb') as fin:
    return pickle.load(fin)


def fix_random_seeds(seed):
  tf.random.set_seed(seed)
  np.random.seed(seed)


def get_loss_fn(loss_name) -> tf.keras.losses.Loss:
  if loss_name == 'mse':
    return tf.keras.losses.MeanSquaredError()
  elif loss_name == 'mae':
    return tf.keras.losses.MeanAbsoluteError()
  elif loss_name == 'bce':
    return tf.keras.losses.BinaryCrossentropy()
  else:
    raise ValueError(f'Loss with name {loss_name} is not supported.')


@tf.function
def compute_true_acc(predictions):
  predictions_true = tf.greater(predictions, 0.5)
  predictions_true = tf.cast(predictions_true, predictions.dtype)
  return tf.reduce_sum(predictions_true) / tf.size(predictions_true, out_type=predictions.dtype)


@tf.function
def compute_fake_acc(predictions):
  predictions_fake = tf.less(predictions, 0.5)
  predictions_fake = tf.cast(predictions_fake, predictions.dtype)
  return tf.reduce_sum(predictions_fake) / tf.size(predictions_fake, out_type=predictions.dtype)


@tf.function
def compute_kl(mu):
  mu_2 = tf.pow(mu, 2)
  encoding_loss = tf.reduce_mean(mu_2)
  return encoding_loss
