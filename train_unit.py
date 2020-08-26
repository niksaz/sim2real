# Author: Mikita Sazanovich

import os
import math
import pickle

import tensorflow as tf
import numpy as np
import imlib
import tf2lib
import tqdm

import configuration
import data
import layers
import optimization
import utils


@tf.function
def _compute_true_acc(predictions):
  predictions_true = tf.greater(predictions, 0.5)
  predictions_true = tf.cast(predictions_true, predictions.dtype)
  return tf.reduce_sum(predictions_true) / tf.size(predictions_true, out_type=predictions.dtype)


@tf.function
def _compute_fake_acc(predictions):
  predictions_fake = tf.less(predictions, 0.5)
  predictions_fake = tf.cast(predictions_fake, predictions.dtype)
  return tf.reduce_sum(predictions_fake) / tf.size(predictions_fake, out_type=predictions.dtype)


@tf.function
def _compute_kl(mu):
  mu_2 = tf.pow(mu, 2)
  encoding_loss = tf.reduce_mean(mu_2)
  return encoding_loss


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
    encoded_a = self.encoder_a(x_a, training=training)
    encoded_b = self.encoder_b(x_b, training=training)
    encoded_ab = tf.concat((encoded_a, encoded_b), axis=0)
    encoded_shared = self.encoder_shared(encoded_ab, training=training)
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
  def joint_train_step(self, images_a, images_b, actions_a):
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
      ad_loss_a = ad_true_loss_a + ad_fake_loss_a
      ad_loss_b = ad_true_loss_b + ad_fake_loss_b
      dis_loss = self.hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b)

      x_bab, shared_ba = self.model.encode_a_decode_b(x_ba, training=training)
      x_aba, shared_ab = self.model.encode_b_decode_a(x_ab, training=training)
      ad_loss_a = self.dis_loss_criterion(y_true=all1, y_pred=out_fake_a)
      ad_loss_b = self.dis_loss_criterion(y_true=all1, y_pred=out_fake_b)
      ll_loss_a = self.ll_loss_criterion(y_true=images_a, y_pred=x_aa)
      ll_loss_b = self.ll_loss_criterion(y_true=images_b, y_pred=x_bb)
      ll_loss_aba = self.ll_loss_criterion(y_true=images_a, y_pred=x_aba)
      ll_loss_bab = self.ll_loss_criterion(y_true=images_b, y_pred=x_bab)
      shared_a, shared_b = tf.split(shared, num_or_size_splits=[len(images_a), len(images_b)], axis=0)
      kl_direct_a_loss = _compute_kl(shared_a)
      kl_direct_b_loss = _compute_kl(shared_b)
      kl_cycle_ab_loss = _compute_kl(shared_ab)
      kl_cycle_ba_loss = _compute_kl(shared_ba)
      z_recon_loss_a = self.z_recon_loss_criterion(shared_a, shared_ab)
      z_recon_loss_b = self.z_recon_loss_criterion(shared_b, shared_ba)
      predictions_a = self.controller(shared_a, training=training)
      control_loss_a = self.control_loss_criterion(actions_a, predictions_a)
      predictions_ab = self.controller(shared_ab, training=training)
      control_loss_ab = self.control_loss_criterion(actions_a, predictions_ab)
      gen_loss = (
          self.hyperparameters['control_w'] * (control_loss_a + control_loss_ab)
          + self.hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b)
          + self.hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab)
          + self.hyperparameters['kl_direct_link_w'] * (kl_direct_a_loss + kl_direct_b_loss)
          + self.hyperparameters['kl_cycle_link_w'] * (kl_cycle_ab_loss + kl_cycle_ba_loss)
          + self.hyperparameters['z_recon_w'] * (z_recon_loss_a + z_recon_loss_b)
          + self.hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b))

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

    true_a_acc_batch = _compute_true_acc(out_true_a)
    true_b_acc_batch = _compute_true_acc(out_true_b)
    fake_a_acc_batch = _compute_fake_acc(out_fake_a)
    fake_b_acc_batch = _compute_fake_acc(out_fake_b)
    D_loss_dict = {
        'true_a_acc_batch': true_a_acc_batch,
        'true_b_acc_batch': true_b_acc_batch,
        'fake_a_acc_batch': fake_a_acc_batch,
        'fake_b_acc_batch': fake_b_acc_batch,
        'loss': dis_loss,
    }
    G_images = [x_aa, x_ba, x_ab, x_bb, x_aba, x_bab]
    G_loss_dict = {
        'kl_direct_a_loss': kl_direct_a_loss,
        'kl_direct_b_loss': kl_direct_b_loss,
        'kl_cycle_ab_loss': kl_cycle_ab_loss,
        'kl_cycle_ba_loss': kl_cycle_ba_loss,
        'ad_loss_a': ad_loss_a,
        'ad_loss_b': ad_loss_b,
        'll_loss_a': ll_loss_a,
        'll_loss_b': ll_loss_b,
        'll_loss_aba': ll_loss_aba,
        'll_loss_bab': ll_loss_bab,
        'loss': gen_loss,
    }
    control_loss_name = self.hyperparameters['control']['loss']
    C_loss_dict = {
        f'train_{control_loss_name}_loss_a': control_loss_a,
        f'train_{control_loss_name}_loss_ab': control_loss_ab,
        f'train_{control_loss_name}_loss': control_loss,
    }
    return D_loss_dict, G_images, G_loss_dict, C_loss_dict

  @staticmethod
  def _compute_and_apply_gradients(models, optimizer, tape, loss) -> None:
    variables = []
    for model in models:
      variables.extend(model.trainable_variables)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))


def compile_sample_paths(dataset_path, descriptor_filename):
  assert descriptor_filename.endswith('.pickle')
  descriptor_path = os.path.join(dataset_path, descriptor_filename)
  with open(descriptor_path, 'rb') as fin:
    sample_files = pickle.load(fin)
  sample_paths = [
      (os.path.join(dataset_path, ifile), os.path.join(dataset_path, afile))
      for ifile, afile in sample_files]
  return sample_paths


def create_image_action_dataset(config, label):
  config_datasets = config['datasets']
  batch_size = config['hyperparameters']['batch_size']
  dataset_path = os.path.join(
      config_datasets['general']['datasets_dir'], config_datasets[label]['dataset_path'])
  train_paths = compile_sample_paths(dataset_path, 'train_files.pickle')
  test_paths = compile_sample_paths(dataset_path, 'test_files.pickle')
  print(f'There are {len(train_paths)} train and {len(test_paths)} test samples in the dataset {label}.')
  load_size = config_datasets['general']['load_size']
  crop_size = config_datasets['general']['crop_size']

  train_image_dataset = data.create_image_dataset(
      [ipath for ipath, _ in train_paths], load_size, crop_size, training=True)
  train_action_dataset = data.create_action_dataset(
      [apath for _, apath in train_paths])
  train_dataset = tf.data.Dataset.zip((train_image_dataset, train_action_dataset))
  shuffle_buffer_size = max(batch_size * 128, 2048)
  train_dataset = train_dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
  train_dataset = train_dataset.repeat(None)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=False)

  test_image_dataset = data.create_image_dataset(
      [ipath for ipath, _ in test_paths], load_size, crop_size, training=False)
  test_action_dataset = data.create_action_dataset(
      [apath for _, apath in test_paths])
  test_dataset = tf.data.Dataset.zip((test_image_dataset, test_action_dataset))
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  test_dataset_len = math.ceil(len(test_paths) / batch_size)

  return train_dataset, test_dataset, test_dataset_len


def main_loop(trainer, datasets, test_iterations, config, checkpoint, samples_dir):
  (a_train_dataset, a_test_dataset), (b_train_dataset, b_test_dataset) = datasets
  optimizer_iterations = config['hyperparameters']['iterations']
  c_loss_mean_dict = {}
  a_dataset_iter = iter(a_train_dataset)
  b_dataset_iter = iter(b_train_dataset)
  for iterations in tqdm.tqdm(range(1, optimizer_iterations + 1)):
    images_a, actions_a = next(a_dataset_iter)
    images_b, _ = next(b_dataset_iter)

    # Training ops
    D_loss_dict, G_images, G_loss_dict, C_loss_dict = trainer.joint_train_step(images_a, images_b, actions_a)
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
    if iterations % config['checkpoint_save_iterations'] == 0 or iterations == optimizer_iterations:
      C_loss_dict = test_model(
          trainer.model, trainer.controller, a_test_dataset, b_test_dataset, test_iterations, config, samples_dir)
      tf2lib.summary(C_loss_dict, step=iterations, name='controller')
      checkpoint.save(iterations)


def test_model(unit_model, controller, a_test_dataset, b_test_dataset, max_iterations, config, samples_dir):
  training = False
  mae_loss_fn = utils.get_loss_fn('mae')
  mae_loss_mean_a = tf.keras.metrics.Mean()
  mae_loss_mean_b = tf.keras.metrics.Mean()
  mse_loss_fn = utils.get_loss_fn('mse')
  mse_loss_mean_a = tf.keras.metrics.Mean()
  mse_loss_mean_b = tf.keras.metrics.Mean()
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
    for shared_x, actions_x, mae_loss_mean_x, mse_loss_mean_x in [
        (shared_a, actions_a, mae_loss_mean_a, mse_loss_mean_a),
        (shared_b, actions_b, mae_loss_mean_b, mse_loss_mean_b)]:
      if shared_x is None:
        continue
      predictions_x = controller(shared_x, training=training)
      mae_loss = mae_loss_fn(actions_x, predictions_x)
      mae_loss_mean_x.update_state(mae_loss.numpy())
      mse_loss = mse_loss_fn(actions_x, predictions_x)
      mse_loss_mean_x.update_state(mse_loss.numpy())

  C_loss_dict = {
      'test_mae_loss_a': mae_loss_mean_a.result(),
      'test_mae_loss_b': mae_loss_mean_b.result(),
      'test_mse_loss_a': mse_loss_mean_a.result(),
      'test_mse_loss_b': mse_loss_mean_b.result(),
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
  checkpoint = tf2lib.Checkpoint(checkpoint_dict, checkpoints_dir, max_to_keep=5)
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
