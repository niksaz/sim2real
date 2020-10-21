# Author: Mikita Sazanovich

import os
import math
import multiprocessing
import pickle

import tensorflow as tf
import numpy as np
import imlib
import tf2lib
import tqdm

import configuration
import data
import layers
import metrics
import optimization
import utils


class UNITModel(object):
  def __init__(self, config):
    gen_hyperparameters = config['hyperparameters']['gen']
    self.encoder_a = layers.Encoder(gen_hyperparameters)
    self.encoder_b = layers.Encoder(gen_hyperparameters)
    self.encoder_shared = layers.EncoderShared(gen_hyperparameters)
    self.decoder_shared = layers.DecoderShared(gen_hyperparameters)
    self.decoder_a = layers.Decoder(gen_hyperparameters)
    self.decoder_b = layers.Decoder(gen_hyperparameters)

    dis_hyperparameters = config['hyperparameters']['dis']
    self.dis_a = layers.Discriminator(dis_hyperparameters)
    self.dis_b = layers.Discriminator(dis_hyperparameters)

  @tf.function
  def encode_ab_decode_aabb(self, x_a, x_b, training):
    encoded_shared = self.encode_ab(x_a, x_b, training)
    decoded_shared = self.decoder_shared(encoded_shared, training=training)
    decoded_a = self.decoder_a(decoded_shared, training=training)
    decoded_b = self.decoder_b(decoded_shared, training=training)
    x_aa, x_ba = tf.split(decoded_a, num_or_size_splits=[len(x_a), len(x_b)], axis=0)
    x_ab, x_bb = tf.split(decoded_b, num_or_size_splits=[len(x_a), len(x_b)], axis=0)
    return x_aa, x_ba, x_ab, x_bb, encoded_shared

  @tf.function
  def encode_a_decode_b(self, x_a, training):
    encoded_a = self.encoder_a(x_a, training=training)
    encoded_shared = self.encoder_shared(encoded_a, training=training)
    decoded_shared = self.decoder_shared(encoded_shared, training=training)
    decoded_b = self.decoder_b(decoded_shared, training=training)
    return decoded_b, encoded_shared

  @tf.function
  def encode_b_decode_a(self, x_b, training):
    encoded_b = self.encoder_b(x_b, training=training)
    encoded_shared = self.encoder_shared(encoded_b, training=training)
    decoded_shared = self.decoder_shared(encoded_shared, training=training)
    decoded_a = self.decoder_a(decoded_shared, training=training)
    return decoded_a, encoded_shared

  @tf.function
  def encode_ab(self, x_a, x_b, training):
    encoded_a = self.encoder_a(x_a, training=training)
    encoded_b = self.encoder_b(x_b, training=training)
    encoded_ab = tf.concat((encoded_a, encoded_b), axis=0)
    encoded_shared = self.encoder_shared(encoded_ab, training=training)
    return encoded_shared


class Trainer(object):
  def __init__(self, model, controller, hyperparameters):
    super(Trainer, self).__init__()
    self.model = model
    self.controller = controller
    self.hyperparameters = hyperparameters
    self.gen_opt = optimization.create_optimizer_from_params(hyperparameters['gen']['optimizer'])
    self.dis_opt = optimization.create_optimizer_from_params(hyperparameters['dis']['optimizer'])
    self.control_opt = optimization.create_optimizer_from_params(hyperparameters['control']['optimizer'])
    self.dis_loss_criterion = utils.get_loss_fn('bce')
    self.ll_loss_criterion = utils.get_loss_fn('mae')
    self.z_recon_loss_criterion = utils.get_loss_fn('mae')
    self.control_loss_criterion = utils.get_loss_fn(hyperparameters['control']['loss'])

  @tf.function
  def joint_train_step(self, images_a, following_images_a, images_b, following_images_b, actions_a):
    training = True
    with tf.GradientTape(persistent=True) as t:
      x_aa, x_ba, x_ab, x_bb, shared = self.model.encode_ab_decode_aabb(images_a, images_b, training=training)
      data_a = tf.concat((images_a, x_ba), axis=0)
      data_b = tf.concat((images_b, x_ab), axis=0)
      res_a = self.model.dis_a(data_a, training=training)
      res_b = self.model.dis_b(data_b, training=training)
      out_a = tf.keras.activations.sigmoid(res_a)
      out_b = tf.keras.activations.sigmoid(res_b)
      out_true_a, out_fake_a = tf.split(out_a, num_or_size_splits=[len(images_a), len(x_ba)], axis=0)
      out_true_b, out_fake_b = tf.split(out_b, num_or_size_splits=[len(images_b), len(x_ab)], axis=0)
      all1 = tf.ones_like(out_true_a)
      all0 = tf.zeros_like(out_true_a)
      ad_true_loss_a = self.dis_loss_criterion(y_true=all1, y_pred=out_true_a)
      ad_true_loss_b = self.dis_loss_criterion(y_true=all1, y_pred=out_true_b)
      ad_fake_loss_a = self.dis_loss_criterion(y_true=all0, y_pred=out_fake_a)
      ad_fake_loss_b = self.dis_loss_criterion(y_true=all0, y_pred=out_fake_b)
      dis_ad_loss_a = ad_true_loss_a + ad_fake_loss_a
      dis_ad_loss_b = ad_true_loss_b + ad_fake_loss_b
      dis_loss = self.hyperparameters['gan_w'] * (dis_ad_loss_a + dis_ad_loss_b)

      x_bab, shared_ba = self.model.encode_a_decode_b(x_ba, training=training)
      x_aba, shared_ab = self.model.encode_b_decode_a(x_ab, training=training)
      gen_ad_loss_a = self.dis_loss_criterion(y_true=all1, y_pred=out_fake_a)
      gen_ad_loss_b = self.dis_loss_criterion(y_true=all1, y_pred=out_fake_b)
      ll_loss_a = self.ll_loss_criterion(y_true=images_a, y_pred=x_aa)
      ll_loss_b = self.ll_loss_criterion(y_true=images_b, y_pred=x_bb)
      ll_loss_aba = self.ll_loss_criterion(y_true=images_a, y_pred=x_aba)
      ll_loss_bab = self.ll_loss_criterion(y_true=images_b, y_pred=x_bab)
      shared_a, shared_b = tf.split(shared, num_or_size_splits=[len(images_a), len(images_b)], axis=0)
      kl_direct_a_loss = utils.compute_kl(shared_a)
      kl_direct_b_loss = utils.compute_kl(shared_b)
      kl_cycle_ab_loss = utils.compute_kl(shared_ab)
      kl_cycle_ba_loss = utils.compute_kl(shared_ba)
      z_recon_loss_a = self.z_recon_loss_criterion(shared_a, shared_ab)
      z_recon_loss_b = self.z_recon_loss_criterion(shared_b, shared_ba)
      predictions_a = self.controller(shared_a, training=training)
      control_loss_a = self.control_loss_criterion(actions_a, predictions_a)
      predictions_ab = self.controller(shared_ab, training=training)
      control_loss_ab = self.control_loss_criterion(actions_a, predictions_ab)

      following_ab_shared = self.model.encode_ab(following_images_a, following_images_b, training=training)
      following_a_shared, following_b_shared = tf.split(
          following_ab_shared, num_or_size_splits=[len(following_images_a), len(following_images_b)], axis=0)

      mse_none = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
      following_a_shared_shuffled = tf.gather(
          following_a_shared,
          tf.random.shuffle(tf.range(tf.shape(following_a_shared)[0])))
      following_a_unbound = mse_none(shared_a, following_a_shared) - mse_none(shared_a, following_a_shared_shuffled)
      z_temporal_loss_a = tf.reduce_mean(tf.maximum(tf.zeros_like(following_a_unbound), following_a_unbound))
      following_b_shared_shuffled = tf.gather(
          following_b_shared,
          tf.random.shuffle(tf.range(tf.shape(following_b_shared)[0])))
      following_b_unbound = mse_none(shared_b, following_b_shared) - mse_none(shared_b, following_b_shared_shuffled)
      z_temporal_loss_b = tf.reduce_mean(tf.maximum(tf.zeros_like(following_b_unbound), following_b_unbound))
      gen_loss = (
          self.hyperparameters['control_w'] * (control_loss_a + control_loss_ab)
          + self.hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b)
          + self.hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab)
          + self.hyperparameters['kl_direct_link_w'] * (kl_direct_a_loss + kl_direct_b_loss)
          + self.hyperparameters['kl_cycle_link_w'] * (kl_cycle_ab_loss + kl_cycle_ba_loss)
          + self.hyperparameters['z_recon_w'] * (z_recon_loss_a + z_recon_loss_b)
          + self.hyperparameters['z_temporal_w'] * (z_temporal_loss_a + z_temporal_loss_b)
          + self.hyperparameters['gan_w'] * (gen_ad_loss_a + gen_ad_loss_b))

      control_loss = self.hyperparameters['control_w'] * (control_loss_a + control_loss_ab)

    dis_models = [self.model.dis_a, self.model.dis_b]
    dis_variables = [v for model in dis_models for v in model.trainable_variables]
    dis_grads = t.gradient(dis_loss, dis_variables)
    gen_models = [
        self.model.encoder_a, self.model.encoder_b,
        self.model.encoder_shared, self.model.decoder_shared,
        self.model.decoder_a, self.model.decoder_b]
    gen_variables = [v for model in gen_models for v in model.trainable_variables]
    gen_grads = t.gradient(gen_loss, gen_variables)
    control_models = [self.controller]
    control_variables = [v for model in control_models for v in model.trainable_variables]
    control_grads = t.gradient(control_loss, control_variables)

    self.dis_opt.apply_gradients(zip(dis_grads, dis_variables))
    self.gen_opt.apply_gradients(zip(gen_grads, gen_variables))
    self.control_opt.apply_gradients(zip(control_grads, control_variables))

    true_a_acc_batch = utils.compute_true_acc(out_true_a)
    true_b_acc_batch = utils.compute_true_acc(out_true_b)
    fake_a_acc_batch = utils.compute_fake_acc(out_fake_a)
    fake_b_acc_batch = utils.compute_fake_acc(out_fake_b)
    D_loss_dict = {
        'true_a_acc_batch': true_a_acc_batch,
        'true_b_acc_batch': true_b_acc_batch,
        'fake_a_acc_batch': fake_a_acc_batch,
        'fake_b_acc_batch': fake_b_acc_batch,
        'gan': self.hyperparameters['gan_w'] * (dis_ad_loss_a + dis_ad_loss_b),
        'loss': dis_loss,
    }
    G_images = [x_aa, x_ba, x_ab, x_bb, x_aba, x_bab]
    G_loss_dict = {
        'control': self.hyperparameters['control_w'] * (control_loss_a + control_loss_ab),
        'll_direct_link': self.hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b),
        'll_cycle_link': self.hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab),
        'kl_direct_link': self.hyperparameters['kl_direct_link_w'] * (kl_direct_a_loss + kl_direct_b_loss),
        'kl_cycle_link': self.hyperparameters['kl_cycle_link_w'] * (kl_cycle_ab_loss + kl_cycle_ba_loss),
        'z_recon': self.hyperparameters['z_recon_w'] * (z_recon_loss_a + z_recon_loss_b),
        'z_temporal': self.hyperparameters['z_temporal_w'] * (z_temporal_loss_a + z_temporal_loss_b),
        'gan': self.hyperparameters['gan_w'] * (gen_ad_loss_a + gen_ad_loss_b),
        'loss': gen_loss,
    }
    C_loss_dict = {
        'control': self.hyperparameters['control_w'] * (control_loss_a + control_loss_ab),
        'loss': control_loss,
    }
    return D_loss_dict, G_images, G_loss_dict, C_loss_dict

  @staticmethod
  def _compute_and_apply_gradients(models, optimizer, tape, loss) -> None:
    variables = []
    for model in models:
      variables.extend(model.trainable_variables)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))


def create_image_action_dataset_from_paths(dataset_path, train_episodes, test_episodes, dataset_gen_cfg, batch_size):
  def get_parse_img_fn(training):
    @tf.function
    def parse_img(path):
      img = tf.io.read_file(path)
      img = tf.image.decode_png(img, 3)  # fix channels to 3
      img_preprocessing_fn = data.img_preprocessing_fn(
          dataset_gen_cfg['load_size'],
          dataset_gen_cfg['crop_size'],
          training=training)
      img = img_preprocessing_fn(img)
      return img
    return parse_img
  parse_img_train = get_parse_img_fn(training=True)
  parse_img_test = get_parse_img_fn(training=False)

  @tf.function
  def parse_action(path):
    action = tf.py_function(lambda path: np.load(path.numpy()), inp=[path], Tout=tf.float32)
    return action

  train_frames = [(bounds, i) for bounds in train_episodes for i in range(bounds[0], bounds[1])]
  def train_generator():
    while True:
      np.random.shuffle(train_frames)
      for bounds, i in train_frames:
        neighs = [i + 1 if i + 1 < bounds[1] else i]
        neigh = np.random.choice(neighs, 1)[0]
        ipath = os.path.join(dataset_path, f'{i}.png')
        npath = os.path.join(dataset_path, f'{neigh}.png')
        apath = os.path.join(dataset_path, f'{i}.npy')
        yield ipath, npath, apath

  test_frames = [(bounds, i) for bounds in test_episodes for i in range(bounds[0], bounds[1])]
  def test_generator():
    for bounds, i in test_frames:
      ipath = os.path.join(dataset_path, f'{i}.png')
      apath = os.path.join(dataset_path, f'{i}.npy')
      yield ipath, apath

  @tf.function
  def parse_train_data(ipath, ifpath, apath):
    return parse_img_train(ipath), parse_img_train(ifpath), parse_action(apath)

  @tf.function
  def parse_test_data(ipath, apath):
    return parse_img_test(ipath), parse_action(apath)

  n_map_threads = multiprocessing.cpu_count()

  train_dataset = tf.data.Dataset.from_generator(
    generator=train_generator,
    output_types=(tf.string, tf.string, tf.string),
    output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
  )
  train_dataset = train_dataset.map(parse_train_data, num_parallel_calls=n_map_threads)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=False)

  test_dataset = tf.data.Dataset.from_generator(
      generator=test_generator,
      output_types=(tf.string, tf.string),
      output_shapes=(tf.TensorShape([]), tf.TensorShape([])),
  )
  test_dataset = test_dataset.map(parse_test_data, num_parallel_calls=n_map_threads)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  test_dataset_len = math.ceil(len(test_frames) / batch_size)

  return train_dataset, test_dataset, test_dataset_len


def load_pickle_fin(pickle_path):
  with open(pickle_path, 'rb') as fin:
    return pickle.load(fin)


def compile_sample_paths(dataset_path, descriptor_filename):
  assert descriptor_filename.endswith('.pickle')
  descriptor_path = os.path.join(dataset_path, descriptor_filename)
  sample_files = load_pickle_fin(descriptor_path)
  sample_paths = []
  for files_set in sample_files:
    paths_set = [os.path.join(dataset_path, file) for file in files_set]
    sample_paths.append(paths_set)
  return sample_paths


def create_image_action_dataset(config, label):
  config_datasets = config['datasets']
  batch_size = config['hyperparameters']['batch_size']
  dataset_path = os.path.join(
      config_datasets['general']['datasets_dir'], config_datasets[label]['dataset_path'])
  train_episodes = load_pickle_fin(os.path.join(dataset_path, 'train_episodes.pickle'))
  test_episodes = load_pickle_fin(os.path.join(dataset_path, 'test_episodes.pickle'))
  print(f'There are {len(train_episodes)} train and {len(test_episodes)} test episodes in the dataset {label}.')
  return create_image_action_dataset_from_paths(
      dataset_path, train_episodes, test_episodes, config_datasets['general'], batch_size)


def main_loop(trainer, datasets, test_iterations, config, checkpoint, samples_dir):
  (a_train_dataset, a_test_dataset), (b_train_dataset, b_test_dataset) = datasets
  optimizer_iterations = config['hyperparameters']['iterations']
  c_loss_mean_dict = {}
  a_dataset_iter = iter(a_train_dataset)
  b_dataset_iter = iter(b_train_dataset)
  for iterations in tqdm.tqdm(range(1, optimizer_iterations + 1)):
    images_a, following_images_a, actions_a = next(a_dataset_iter)
    images_b, following_images_b, _ = next(b_dataset_iter)

    # Training ops
    D_loss_dict, G_images, G_loss_dict, C_loss_dict = trainer.joint_train_step(
        images_a, following_images_a, images_b, following_images_b, actions_a)
    for c_loss_label, c_loss in C_loss_dict.items():
      if c_loss_label not in c_loss_mean_dict:
        c_loss_mean_dict[c_loss_label] = tf.keras.metrics.Mean()
      c_loss_mean_dict[c_loss_label].update_state(c_loss.numpy())

    # Logging ops
    if iterations % config['log_iterations'] == 0:
      for c_loss_label, c_loss_mean in c_loss_mean_dict.items():
        C_loss_dict[c_loss_label] = c_loss_mean.result()
        c_loss_mean.reset_states()
      tf2lib.summary(D_loss_dict, step=iterations, name='discriminator')
      tf2lib.summary(G_loss_dict, step=iterations, name='generator')
      tf2lib.summary(C_loss_dict, step=iterations, name='controller')
    # Displaying ops
    if iterations % config['image_save_iterations'] == 0:
      img_filename = os.path.join(samples_dir, f'train_{iterations}.jpg')
    elif iterations % config['image_display_iterations'] == 0:
      img_filename = os.path.join(samples_dir, f'train.jpg')
    else:
      img_filename = None
    if img_filename:
      img = imlib.immerge(np.concatenate([images_a, images_b] + G_images, axis=0), n_rows=8)
      imlib.imwrite(img, img_filename)
    # Testing and checkpointing ops
    if iterations % config['test_every_iterations'] == 0 or iterations == optimizer_iterations:
      C_loss_dict = test_model(
          trainer.model, trainer.controller, a_test_dataset, b_test_dataset, test_iterations, samples_dir)
      tf2lib.summary(C_loss_dict, step=iterations, name='controller')
      checkpoint.save(iterations)


def test_model(unit_model, controller, a_test_dataset, b_test_dataset, max_iterations, samples_dir):
  training = False
  mae_metric_a = metrics.MAEMetric()
  mae_metric_b = metrics.MAEMetric()
  mse_metric_a = metrics.MSEMetric()
  mse_metric_b = metrics.MSEMetric()
  bmae_metric_a = metrics.BMAEMetric()
  bmae_metric_b = metrics.BMAEMetric()
  bmse_metric_a = metrics.BMSEMetric()
  bmse_metric_b = metrics.BMSEMetric()
  a_test_iter = iter(a_test_dataset)
  b_test_iter = iter(b_test_dataset)
  for iterations in tqdm.tqdm(range(1, max_iterations + 1)):
    try:
      images_a, actions_a = next(a_test_iter)
    except StopIteration:
      images_a, actions_a = None, None
    try:
      images_b, actions_b = next(b_test_iter)
    except StopIteration:
      images_b, actions_b = None, None

    # Inference ops
    G_images = None
    if images_a is not None and images_b is not None:
      x_aa, x_ba, x_ab, x_bb, shared = unit_model.encode_ab_decode_aabb(images_a, images_b, training=training)
      x_bab, _ = unit_model.encode_a_decode_b(x_ba, training=training)
      x_aba, _ = unit_model.encode_b_decode_a(x_ab, training=training)
      G_images = [x_aa, x_ba, x_ab, x_bb, x_aba, x_bab]
      shared_a, shared_b = tf.split(shared, num_or_size_splits=[len(images_a), len(images_b)], axis=0)
    elif images_a is not None:
      encoded_a = unit_model.encoder_a(images_a, training=training)
      shared_a = unit_model.encoder_shared(encoded_a, training=training)
      shared_b = None
    elif images_b is not None:
      encoded_b = unit_model.encoder_b(images_b, training=training)
      shared_b = unit_model.encoder_shared(encoded_b, training=training)
      shared_a = None
    else:
      raise AssertionError('There are no images either from A or B during the test.')

    # Displaying ops
    if iterations % (max_iterations // 10) == 0:
      img_filename = os.path.join(samples_dir, f'test_{iterations}.jpg')
      images = []
      if images_a is not None:
        images.append(images_a)
      if images_b is not None:
        images.append(images_b)
      if G_images is not None:
        images.extend(G_images)
      img = imlib.immerge(np.concatenate(images, axis=0), n_rows=len(images))
      imlib.imwrite(img, img_filename)

    # Control loss accumulation
    for shared_x, actions_x, metrics_x in [
        (shared_a, actions_a, (mae_metric_a, mse_metric_a, bmae_metric_a, bmse_metric_a)),
        (shared_b, actions_b, (mae_metric_b, mse_metric_b, bmae_metric_b, bmse_metric_b)),
    ]:
      if shared_x is None:
        continue
      predictions_x = controller(shared_x, training=training)
      actions_x = actions_x.numpy()
      predictions_x = predictions_x.numpy()
      for metric_x in metrics_x:
        metric_x.update_state(actions_x, predictions_x)

  C_loss_dict = {
      'test_mae_metric_a': mae_metric_a.result(),
      'test_mse_metric_a': mse_metric_a.result(),
      'test_bmae_metric_a': bmae_metric_a.result(),
      'test_bmse_metric_a': bmse_metric_a.result(),
      'test_mae_metric_b': mae_metric_b.result(),
      'test_mse_metric_b': mse_metric_b.result(),
      'test_bmae_metric_b': bmae_metric_b.result(),
      'test_bmse_metric_b': bmse_metric_b.result(),
  }
  return C_loss_dict


def main():
  args = configuration.parse_args()
  config = configuration.load_config(args.config_path)
  print('args:', args)
  print('config:', config)

  utils.fix_random_seeds(config['hyperparameters']['seed'])

  a_train_dataset, a_test_dataset, a_test_length = create_image_action_dataset(config, 'domain_a')
  b_train_dataset, b_test_dataset, b_test_length = create_image_action_dataset(config, 'domain_b')

  unit_model = UNITModel(config)
  gen_hyperparameters = config['hyperparameters']['gen']
  z_ch = gen_hyperparameters['ch'] * 2 ** (gen_hyperparameters['n_enc_front_blk'] - 1)
  control_hyperparameters = config['hyperparameters']['control']
  controller = layers.Controller(z_ch, control_hyperparameters, 2)
  trainer = Trainer(unit_model, controller, config['hyperparameters'])

  output_dir, (samples_dir, summaries_dir, checkpoints_dir) = utils.create_output_dirs(
      args.output_dir_base, 'unit', args.tag, ['samples', 'summaries', 'checkpoints'])
  configuration.dump_config(config, os.path.join(output_dir, 'config.yaml'))

  checkpoint_dict = {
      'encoder_a': unit_model.encoder_a,
      'encoder_b': unit_model.encoder_b,
      'encoder_shared': unit_model.encoder_shared,
      'decoder_shared': unit_model.decoder_shared,
      'decoder_a': unit_model.decoder_a,
      'decoder_b': unit_model.decoder_b,
      'dis_a': unit_model.dis_a,
      'dis_b': unit_model.dis_b,
      'controller': controller.model,
      'gen_opt': trainer.gen_opt,
      'dis_opt': trainer.dis_opt,
      'control_opt': trainer.control_opt,
  }
  checkpoint = tf2lib.Checkpoint(checkpoint_dict, checkpoints_dir, max_to_keep=1)
  try:  # Restore checkpoint
    checkpoint.restore().assert_existing_objects_matched()
  except Exception as e:
    print(e)
  summary_writer = tf.summary.create_file_writer(summaries_dir)

  with summary_writer.as_default():
    datasets = [(a_train_dataset, a_test_dataset), (b_train_dataset, b_test_dataset)]
    test_iterations = max(a_test_length, b_test_length)
    main_loop(trainer, datasets, test_iterations, config, checkpoint, samples_dir)

  if args.summarize:
    unit_model.encoder_a.model.summary()
    unit_model.encoder_b.model.summary()
    unit_model.encoder_shared.model.summary()
    unit_model.decoder_shared.model.summary()
    controller.model.summary()


if __name__ == '__main__':
  main()
