# Author: Mikita Sazanovich

import os
import multiprocessing

import numpy as np
import tensorflow as tf
import tf2lib as tl
import pylib


def create_image_dataset(config_datasets, dataset_label, training):
  datasets_dir = config_datasets['general']['datasets_dir']
  load_size = config_datasets['general']['load_size']
  crop_size = config_datasets['general']['crop_size']
  n_map_threads = multiprocessing.cpu_count()

  def _parse_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, 3)  # fix channels to 3
    return (img, )

  _preprocess_img = img_preprocessing_fn(load_size, crop_size, training)

  def _map_img(*args):
    return _preprocess_img(*_parse_img(*args))

  config_dataset = config_datasets[dataset_label]
  paths_image = sorted(pylib.glob(
      os.path.join(datasets_dir, config_dataset['dataset_name']), config_dataset['filter_images']))
  paths_count = len(paths_image)
  image_dataset = tf.data.Dataset.from_tensor_slices(paths_image)
  image_dataset = image_dataset.map(_map_img, num_parallel_calls=n_map_threads)
  return image_dataset, paths_count


def create_action_dataset(config_datasets, dataset_label):
  datasets_dir = config_datasets['general']['datasets_dir']
  n_map_threads = multiprocessing.cpu_count()

  def _parse_npy(path):
    np_array = np.load(path.numpy())
    np_array = np_array.astype(np.float32)
    return (np_array,)

  config_action = config_datasets[dataset_label]
  paths_action = sorted(pylib.glob(
      os.path.join(datasets_dir, config_action['dataset_name']), config_action['filter_actions']))
  paths_count = len(paths_action)
  action_dataset = tf.data.Dataset.from_tensor_slices(paths_action)
  action_dataset = action_dataset.map(
      lambda path: tf.py_function(_parse_npy, inp=[path], Tout=tf.float32),
      num_parallel_calls=n_map_threads)
  return action_dataset, paths_count


def img_preprocessing_fn(load_size, crop_size, training):
  @tf.function
  def _map_fn(img):  # preprocessing
    if tf.shape(img)[1] / tf.shape(img)[0] == 4 / 3:
      # Remove the top third of the image, since it is an unprocessed image.
      img = img[tf.shape(img)[0] // 3:, :, :]

    if training:
      img = tf.image.resize(img, load_size)
      img = tf.image.random_crop(img, crop_size + [tf.shape(img)[-1]])
    else:
      img = tf.image.resize(img, crop_size)  # or img = tf.image.resize(img, load_size); img = tl.center_crop(img, crop_size)

    img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
    img = img * 2 - 1  # or img = tf.image.rgb_to_yuv(img)
    return img
  return _map_fn


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    _map_fn = img_preprocessing_fn(load_size, crop_size, training)
    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)
