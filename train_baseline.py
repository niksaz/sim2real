# Author: Mikita Sazanovich

import pylib
import os
import multiprocessing
import numpy as np
import tensorflow as tf

import utils
import layers
import tqdm


def create_dataset(config, dataset_label, training):
  config_datasets = config['datasets']
  datasets_dir = config_datasets['general']['datasets_dir']
  load_size = config_datasets['general']['load_size']
  crop_size = config_datasets['general']['crop_size']
  config_dataset = config_datasets[dataset_label]
  batch_size = config['hyperparameters']['batch_size']
  n_map_threads = multiprocessing.cpu_count()

  # Image dataset
  img_paths = list(sorted(
      pylib.glob(os.path.join(datasets_dir, config_dataset['dataset_name']), config_dataset['filter'])))

  if training:
    @tf.function
    def _map_img(img):  # preprocessing
      img = tf.image.random_flip_left_right(img)
      img = tf.image.resize(img, load_size)
      img = tf.image.random_crop(img, crop_size + [tf.shape(img)[-1]])
      img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
      img = img * 2 - 1
      return img
  else:
    @tf.function
    def _map_img(img):  # preprocessing
      img = tf.image.resize(img,
                            crop_size)  # or img = tf.image.resize(img, load_size); img = tl.center_crop(img, crop_size)
      img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
      img = img * 2 - 1
      return img

  def parse_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, 3)  # fix channels to 3
    return (img, )

  def map_img(*args):
    return _map_img(*parse_img(*args))

  img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
  img_dataset = img_dataset.map(map_img, num_parallel_calls=n_map_threads)

  # Action dataset
  act_paths = list(sorted(
      pylib.glob(os.path.join(datasets_dir, config_dataset['dataset_name']), config_dataset['filter_actions'])))

  def parse_npy(path):
    data = np.load(path.numpy())
    data = data.astype(np.float32)
    return (data,)

  act_dataset = tf.data.Dataset.from_tensor_slices(act_paths)
  act_dataset = act_dataset.map(
      lambda path: tf.py_function(parse_npy, inp=[path], Tout=tf.float32),
      num_parallel_calls=n_map_threads)

  # Image and action dataset
  zip_dataset = tf.data.Dataset.zip((img_dataset, act_dataset))
  if training:
    shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048
    zip_dataset = zip_dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
    zip_dataset = zip_dataset.repeat(None)
    len_dataset = None
  else:
    len_dataset = (max(len(img_paths), len(act_paths)) + batch_size - 1) // batch_size
  zip_dataset = zip_dataset.batch(batch_size, drop_remainder=False)
  return zip_dataset, len_dataset


def train_model(encoder_a, encoder_shared, controller, loss_fn, config):
  training = True
  lr = config['hyperparameters']['lr']
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999, decay=0.0001)

  @tf.function
  def train_step(images, actions):
    with tf.GradientTape() as t:
      z = encoder_a(images, training=training)
      z = encoder_shared(z, training=training)
      predictions = controller(z, training=training)
      loss = loss_fn(actions, predictions)

    variables = []
    model_parts = [encoder_a, encoder_shared, controller]
    for model in model_parts:
      variables.extend(model.trainable_variables)
    grads = t.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

  train_mean_loss = tf.keras.metrics.Mean()
  train_dataset, _ = create_dataset(config, dataset_label='train_a', training=training)
  dataset_iter = iter(train_dataset)
  max_iterations = config['hyperparameters']['max_iterations']
  for iterations in tqdm.tqdm(range(max_iterations)):
    images, actions = next(dataset_iter)
    loss = train_step(images, actions)
    train_mean_loss.update_state(loss.numpy())
    if (iterations + 1) % 1000 == 0 or iterations + 1 == max_iterations:
      print('train_mean_loss is', train_mean_loss.result().numpy())
      train_mean_loss.reset_states()


def test_model(encoder_a, encoder_shared, controller, loss_fn, config):
  training = False

  @tf.function
  def test_step(images, actions):
    z = encoder_a(images, training=training)
    z = encoder_shared(z, training=training)
    predictions = controller(z, training=training)
    loss = loss_fn(actions, predictions)
    return loss

  test_mean_loss = tf.keras.metrics.Mean()
  test_dataset, test_dataset_length = create_dataset(config, dataset_label='test_a', training=training)
  dataset_iter = iter(test_dataset)
  for _ in tqdm.tqdm(range(test_dataset_length)):
    images, actions = next(dataset_iter)
    loss = test_step(images, actions)
    test_mean_loss.update_state(loss.numpy())
  print('test_mean_loss is', test_mean_loss.result().numpy())


def main():
  args = utils.parse_args()
  config = utils.load_config(args.config_path)
  print('args:', args)
  print('config:', config)

  gen_hyperparameters = config['hyperparameters']['gen']
  encoder_a = layers.Encoder(gen_hyperparameters)
  encoder_shared = layers.EncoderShared(gen_hyperparameters)
  z_ch = gen_hyperparameters['ch'] * 2 ** (gen_hyperparameters['n_enc_front_blk'] - 1)
  control_hyperparameters = config['hyperparameters']['control']
  controller = layers.Controller(control_hyperparameters, z_ch)

  loss_fn = tf.keras.losses.MeanSquaredError()
  train_model(encoder_a, encoder_shared, controller, loss_fn, config)
  test_model(encoder_a, encoder_shared, controller, loss_fn, config)


if __name__ == '__main__':
  main()
