# Author: Mikita Sazanovich

import os
import time
import multiprocessing
import numpy as np
import tensorflow as tf
import pylib
import tqdm

import configuration
import data
import layers
import optimization


def create_dataset(config, dataset_label, training):
  config_datasets = config['datasets']
  datasets_dir = config_datasets['general']['datasets_dir']
  load_size = config_datasets['general']['load_size']
  crop_size = config_datasets['general']['crop_size']
  config_dataset = config_datasets[dataset_label]
  dataset_dir = os.path.join(datasets_dir, config_dataset['dataset_name'])
  batch_size = config['hyperparameters']['batch_size']
  n_map_threads = multiprocessing.cpu_count()

  # Image dataset
  img_paths = sorted(pylib.glob(dataset_dir, config_dataset['filter_images']))

  def _parse_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, 3)  # fix channels to 3
    return (img, )

  _preprocess_img = data.img_preprocessing_fn(load_size, crop_size, training)

  def _map_img(*args):
    return _preprocess_img(*_parse_img(*args))

  img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
  img_dataset = img_dataset.map(_map_img, num_parallel_calls=n_map_threads)

  # Action dataset
  act_paths = sorted(pylib.glob(dataset_dir, config_dataset['filter_actions']))

  def _parse_npy(path):
    np_array = np.load(path.numpy())
    np_array = np_array.astype(np.float32)
    return (np_array,)

  act_dataset = tf.data.Dataset.from_tensor_slices(act_paths)
  act_dataset = act_dataset.map(
      lambda path: tf.py_function(_parse_npy, inp=[path], Tout=tf.float32),
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
  optimizer_hyperparameters = config['hyperparameters']['optimizer']
  optimizer = optimization.create_optimizer_from_params(optimizer_hyperparameters)

  @tf.function
  def train_step(images, actions):
    training = True
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

  @tf.function
  def test_step(images, actions):
    training = False
    z = encoder_a(images, training=training)
    z = encoder_shared(z, training=training)
    predictions = controller(z, training=training)
    loss = loss_fn(actions, predictions)
    return loss

  train_mean_loss = tf.keras.metrics.Mean()
  train_dataset, _ = create_dataset(config, dataset_label='train_a', training=True)
  train_iter = iter(train_dataset)
  test_mean_loss = tf.keras.metrics.Mean()
  test_dataset, test_dataset_length = create_dataset(config, dataset_label='test_a', training=False)

  optimizer_iterations = optimizer_hyperparameters['iterations']
  for iterations in tqdm.tqdm(range(optimizer_iterations)):
    images, actions = next(train_iter)
    loss = train_step(images, actions)
    train_mean_loss.update_state(loss.numpy())

    # log the training loss
    if (iterations + 1) % config['log_iterations'] == 0 or iterations + 1 == optimizer_iterations:
      tf.summary.scalar('train_mean_loss', train_mean_loss.result(), step=iterations+1)
      train_mean_loss.reset_states()

    # log the test loss
    if (iterations + 1) % config['checkpoint_save_iterations'] == 0 or iterations + 1 == optimizer_iterations:
      test_iter = iter(test_dataset)
      for _ in range(test_dataset_length):
        images, actions = next(test_iter)
        loss = test_step(images, actions)
        test_mean_loss.update_state(loss.numpy())
      tf.summary.scalar('test_mean_loss', test_mean_loss.result(), step=iterations+1)
      test_mean_loss.reset_states()


def main():
  args = configuration.parse_args()
  config = configuration.load_config(args.config_path)
  print('args:', args)
  print('config:', config)

  gen_hyperparameters = config['hyperparameters']['gen']
  encoder_a = layers.Encoder(gen_hyperparameters)
  encoder_shared = layers.EncoderShared(gen_hyperparameters)
  z_ch = gen_hyperparameters['ch'] * 2 ** (gen_hyperparameters['n_enc_front_blk'] - 1)
  control_hyperparameters = config['hyperparameters']['control']
  controller = layers.Controller(z_ch, control_hyperparameters, 2)

  # output dirs
  output_dir_base = args.output_dir_base
  output_dir_name = f'{args.tag}-{time.strftime("%Y%m%d%H%M%S")}'
  output_dir = os.path.join(output_dir_base, 'baseline', output_dir_name)
  os.makedirs(output_dir, exist_ok=False)
  summaries_dir = os.path.join(output_dir, 'summaries')
  os.makedirs(summaries_dir, exist_ok=False)

  writer = tf.summary.create_file_writer(summaries_dir)
  loss_fn = tf.keras.losses.MeanSquaredError()
  with writer.as_default():
    train_model(encoder_a, encoder_shared, controller, loss_fn, config)

  if args.summarize:
    encoder_a.model.summary()
    encoder_shared.model.summary()
    controller.seq.summary()


if __name__ == '__main__':
  main()
