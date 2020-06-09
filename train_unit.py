# Author: Mikita Sazanovich

import os
import time

import tensorflow as tf
import numpy as np
import pylib
import imlib
import tf2lib
import tqdm

import configuration
import data
import layers


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
  def encoder_decoder_ab_abab(self, x_a, x_b):
    encoded_a = self.encoder_a(x_a)
    encoded_b = self.encoder_b(x_b)
    encoded_ab = tf.concat((encoded_a, encoded_b), axis=0)
    encoded_shared = self.encoder_shared(encoded_ab)
    decoded_shared = self.decoder_shared(encoded_shared)
    decoded_a = self.decoder_a(decoded_shared)
    decoded_b = self.decoder_b(decoded_shared)
    x_aa, x_ba = tf.split(decoded_a, num_or_size_splits=2, axis=0)
    x_ab, x_bb = tf.split(decoded_b, num_or_size_splits=2, axis=0)
    return x_aa, x_ba, x_ab, x_bb, encoded_shared

  @tf.function
  def encoder_decoder_a_b(self, x_a):
    encoded_a = self.encoder_a(x_a)
    encoded_shared = self.encoder_shared(encoded_a)
    decoded_shared = self.decoder_shared(encoded_shared)
    decoded_b = self.decoder_b(decoded_shared)
    return decoded_b, encoded_shared

  @tf.function
  def encoder_decoder_b_a(self, x_b):
    encoded_b = self.encoder_b(x_b)
    encoded_shared = self.encoder_shared(encoded_b)
    decoded_shared = self.decoder_shared(encoded_shared)
    decoded_a = self.decoder_a(decoded_shared)
    return decoded_a, encoded_shared


class Trainer(object):
  def __init__(self, model, hyperparameters):
    super(Trainer, self).__init__()
    self.model = model
    self.hyperparameters = hyperparameters
    lr = self.hyperparameters['lr']
    self.enc_dec_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999, decay=0.0001)
    self.discrim_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999, decay=0.0001)
    self.dis_loss_criterion = tf.keras.losses.BinaryCrossentropy()
    self.ll_loss_criterion_a = tf.keras.losses.MeanAbsoluteError()
    self.ll_loss_criterion_b = tf.keras.losses.MeanAbsoluteError()

  @tf.function
  def train_step(self, images_a, images_b):
    D_loss_dict = self.dis_update(images_a, images_b)
    G_images, G_loss_dict = self.enc_dec_update(images_a, images_b)
    return G_images, G_loss_dict, D_loss_dict

  @tf.function
  def dis_update(self, images_a, images_b):
    with tf.GradientTape() as t:
      x_aa, x_ba, x_ab, x_bb, shared = self.model.encoder_decoder_ab_abab(images_a, images_b)
      data_a = tf.concat((images_a, x_ba), axis=0)
      data_b = tf.concat((images_b, x_ab), axis=0)
      res_a = self.model.dis_a(data_a)
      res_b = self.model.dis_b(data_b)
      for it, (this_a, this_b) in enumerate(zip(res_a, res_b)):
        out_a = tf.keras.activations.sigmoid(this_a)
        out_b = tf.keras.activations.sigmoid(this_b)
        out_true_a, out_fake_a = tf.split(out_a, num_or_size_splits=2, axis=0)
        out_true_b, out_fake_b = tf.split(out_b, num_or_size_splits=2, axis=0)
        out_true_n = out_true_a.shape[0]
        out_fake_n = out_fake_a.shape[0]
        all1 = tf.ones([out_true_n])
        all0 = tf.zeros([out_fake_n])
        ad_true_loss_a = self.dis_loss_criterion(y_true=all1, y_pred=out_true_a)
        ad_true_loss_b = self.dis_loss_criterion(y_true=all1, y_pred=out_true_b)
        ad_fake_loss_a = self.dis_loss_criterion(y_true=all0, y_pred=out_fake_a)
        ad_fake_loss_b = self.dis_loss_criterion(y_true=all0, y_pred=out_fake_b)
        if it == 0:
          ad_loss_a = ad_true_loss_a + ad_fake_loss_a
          ad_loss_b = ad_true_loss_b + ad_fake_loss_b
        else:
          ad_loss_a += ad_true_loss_a + ad_fake_loss_a
          ad_loss_b += ad_true_loss_b + ad_fake_loss_b
      loss = self.hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b)

    variables = []
    dis_models = [self.model.dis_a, self.model.dis_b]
    for model in dis_models:
      variables.extend(model.trainable_variables)
    grads = t.gradient(loss, variables)
    self.discrim_opt.apply_gradients(zip(grads, variables))

    true_a_acc_batch = _compute_true_acc(out_true_a)
    true_b_acc_batch = _compute_true_acc(out_true_b)
    fake_a_acc_batch = _compute_fake_acc(out_fake_a)
    fake_b_acc_batch = _compute_fake_acc(out_fake_b)

    D_loss_dict = {
        'true_a_acc_batch': true_a_acc_batch,
        'true_b_acc_batch': true_b_acc_batch,
        'fake_a_acc_batch': fake_a_acc_batch,
        'fake_b_acc_batch': fake_b_acc_batch,
        'loss': loss,
    }
    return D_loss_dict

  @tf.function
  def enc_dec_update(self, images_a, images_b):
    with tf.GradientTape() as t:
      x_aa, x_ba, x_ab, x_bb, shared = self.model.encoder_decoder_ab_abab(images_a, images_b)
      x_bab, shared_bab = self.model.encoder_decoder_a_b(x_ba)
      x_aba, shared_aba = self.model.encoder_decoder_b_a(x_ab)
      outs_a = self.model.dis_a(x_ba)
      outs_b = self.model.dis_b(x_ab)
      for it, (out_a, out_b) in enumerate(zip(outs_a, outs_b)):
        outputs_a = tf.keras.activations.sigmoid(out_a)
        outputs_b = tf.keras.activations.sigmoid(out_b)
        outputs_n = outputs_a.shape[0]
        all_ones = tf.ones([outputs_n])
        ad_loss_a_add = self.dis_loss_criterion(y_true=all_ones, y_pred=outputs_a)
        ad_loss_b_add = self.dis_loss_criterion(y_true=all_ones, y_pred=outputs_b)
        if it == 0:
          ad_loss_a = ad_loss_a_add
          ad_loss_b = ad_loss_b_add
        else:
          ad_loss_a += ad_loss_a_add
          ad_loss_b += ad_loss_b_add

      enc_loss = _compute_kl(shared)
      enc_bab_loss = _compute_kl(shared_bab)
      enc_aba_loss = _compute_kl(shared_aba)
      ll_loss_a = self.ll_loss_criterion_a(y_true=images_a, y_pred=x_aa)
      ll_loss_b = self.ll_loss_criterion_b(y_true=images_b, y_pred=x_bb)
      ll_loss_aba = self.ll_loss_criterion_a(y_true=images_a, y_pred=x_aba)
      ll_loss_bab = self.ll_loss_criterion_b(y_true=images_b, y_pred=x_bab)
      loss = (
          self.hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b)
          + self.hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b)
          + self.hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab)
          + self.hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss)
          + self.hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss))

    variables = []
    gen_models = [
        self.model.encoder_a, self.model.encoder_b,
        self.model.encoder_shared, self.model.decoder_shared,
        self.model.decoder_a, self.model.decoder_b]
    for model in gen_models:
      variables.extend(model.trainable_variables)
    grads = t.gradient(loss, variables)
    self.enc_dec_opt.apply_gradients(zip(grads, variables))

    G_images = [x_aa, x_ba, x_ab, x_bb, x_aba, x_bab]
    G_loss_dict = {
        'enc_loss': enc_loss,
        'enc_bab_loss': enc_bab_loss,
        'enc_aba_loss': enc_aba_loss,
        'ad_loss_a': ad_loss_a,
        'ad_loss_b': ad_loss_b,
        'll_loss_a': ll_loss_a,
        'll_loss_b': ll_loss_b,
        'll_loss_aba': ll_loss_aba,
        'll_loss_bab': ll_loss_bab,
        'loss': loss,
    }
    return G_images, G_loss_dict


def create_datasets(config):
  config_datasets = config['datasets']
  datasets_dir = config_datasets['general']['datasets_dir']
  load_size = config_datasets['general']['load_size']
  crop_size = config_datasets['general']['crop_size']

  batch_size = config['hyperparameters']['batch_size']

  config_train_a = config_datasets['train_a']
  config_train_b = config_datasets['train_b']
  train_a_paths = pylib.glob(os.path.join(datasets_dir, config_train_a['dataset_name']), config_train_a['filter'])
  train_b_paths = pylib.glob(os.path.join(datasets_dir, config_train_b['dataset_name']), config_train_b['filter'])
  ab_train_dataset, ab_train_length = data.make_zip_dataset(
      train_a_paths,
      train_b_paths,
      batch_size,
      load_size,
      crop_size,
      training=True,
      repeat=False)
  print(f'len(train_a_paths) = {len(train_a_paths)}')
  print(f'len(train_b_paths) = {len(train_b_paths)}')
  print(f'{ab_train_length} batches in AB train')

  config_test_a = config_datasets['test_a']
  config_test_b = config_datasets['test_b']
  test_a_paths = pylib.glob(os.path.join(datasets_dir, config_test_a['dataset_name']), config_test_a['filter'])
  test_b_paths = pylib.glob(os.path.join(datasets_dir, config_test_b['dataset_name']), config_test_b['filter'])
  ab_test_dataset, ab_test_length = data.make_zip_dataset(
      test_a_paths,
      test_b_paths,
      batch_size,
      load_size,
      crop_size,
      training=False,
      repeat=True)
  print(f'len(test_a_paths) = {len(test_a_paths)}')
  print(f'len(test_b_paths) = {len(test_b_paths)}')
  print(f'{ab_test_length} batches in AB test')

  return ab_train_dataset, ab_train_length, ab_test_dataset, ab_test_length


def train(config, summaries_dir, samples_dir, ab_train_dataset, trainer, checkpoint):
  train_summary_writer = tf.summary.create_file_writer(summaries_dir)
  max_iterations = config['hyperparameters']['max_iterations']
  dataset_iter = iter(ab_train_dataset)
  for iterations in tqdm.tqdm(range(max_iterations)):
    try:
      images_a, images_b = next(dataset_iter)
    except StopIteration:
      print('Resetting the iterator...')
      dataset_iter = iter(ab_train_dataset)
      images_a, images_b = next(dataset_iter)

    # Training ops
    G_images, G_loss_dict, D_loss_dict = trainer.train_step(images_a, images_b)

    # Logging ops
    if (iterations + 1) % config['log_iterations'] == 0:
      with train_summary_writer.as_default():
        tf2lib.summary(D_loss_dict, step=iterations, name='discriminator')
        tf2lib.summary(G_loss_dict, step=iterations, name='generator')
    # Displaying ops
    if (iterations + 1) % config['image_save_iterations'] == 0:
      img_filename = os.path.join(samples_dir, f'train_{iterations + 1}.jpg')
    elif (iterations + 1) % config['image_display_iterations'] == 0:
      img_filename = os.path.join(samples_dir, f'train.jpg')
    else:
      img_filename = None
    if img_filename:
      img = imlib.immerge(np.concatenate([images_a, images_b] + G_images, axis=0), n_rows=8)
      imlib.imwrite(img, img_filename)
    # Checkpointing ops
    if (iterations + 1) % config['checkpoint_save_iterations'] == 0 or iterations + 1 == max_iterations:
      checkpoint.save(iterations + 1)


def main():
  args = configuration.parse_args()
  config = configuration.load_config(args.config_path)
  print('args:', args)
  print('config:', config)

  tf.random.set_seed(config['hyperparameters']['seed'])
  np.random.seed(config['hyperparameters']['seed'])

  ab_train_dataset, ab_train_length, ab_test_dataset, ab_test_length = create_datasets(config)

  unit_model = UNITModel(config)
  trainer = Trainer(unit_model, config['hyperparameters'])

  # Output directories
  output_dir_base = args.output_dir_base
  output_dir_name = f'unit-{args.tag}-{time.strftime("%Y%m%d%H%M%S")}'
  output_dir = os.path.join(output_dir_base, output_dir_name)
  os.makedirs(output_dir, exist_ok=False)

  samples_dir = os.path.join(output_dir, 'samples')
  summaries_dir = os.path.join(output_dir, 'summaries')
  os.makedirs(samples_dir, exist_ok=False)
  os.makedirs(summaries_dir, exist_ok=False)

  # Checkpointing setup
  checkpoints_dir = os.path.join(output_dir, 'checkpoints')
  checkpoint_dict = {
      'encoder_a': unit_model.encoder_a,
      'encoder_b': unit_model.encoder_b,
      'encoder_shared': unit_model.encoder_shared,
      'decoder_shared': unit_model.decoder_shared,
      'decoder_a': unit_model.decoder_a,
      'decoder_b': unit_model.decoder_b,
      'dis_a': unit_model.dis_a,
      'dis_b': unit_model.dis_b,
      'enc_dec_opt': trainer.enc_dec_opt,
      'discrim_opt': trainer.discrim_opt,
  }
  checkpoint = tf2lib.Checkpoint(checkpoint_dict, checkpoints_dir, max_to_keep=5)
  try:  # Restore checkpoint
    checkpoint.restore().assert_existing_objects_matched()
  except Exception as e:
    print(e)

  if not args.skip_train:
    train(config, summaries_dir, samples_dir, ab_train_dataset, trainer, checkpoint)

  if not args.skip_test:
    # Restore from the test checkpoint
    test_checkpoint_dir = args.test_checkpoinst_dir if args.test_checkpoint_dir else checkpoints_dir
    test_latest_checkpoint = tf.train.latest_checkpoint(test_checkpoint_dir)
    print('The checkpoint for test is', test_latest_checkpoint)
    if not test_latest_checkpoint:
      raise ValueError('No checkpoint is found for the test stage. Specify it in --test_checkpoint_dir.')

    test_checkpoint = tf.train.Checkpoint(**checkpoint_dict)
    test_checkpoint.restore(test_latest_checkpoint).assert_existing_objects_matched()

    test_batches_to_save = 10
    test_dataset_iter = iter(ab_test_dataset)
    for iterations in tqdm.tqdm(range(ab_test_length)):
      try:
        images_a, images_b = next(test_dataset_iter)
      except StopIteration:
        break

      # Inference ops
      x_aa, x_ba, x_ab, x_bb, shared = unit_model.encoder_decoder_ab_abab(images_a, images_b)
      x_bab, shared_bab = unit_model.encoder_decoder_a_b(x_ba)
      x_aba, shared_aba = unit_model.encoder_decoder_b_a(x_ab)
      G_images = [x_aa, x_ba, x_ab, x_bb, x_aba, x_bab]

      # Displaying ops
      if (iterations + 1) % (ab_test_length // test_batches_to_save) == 0:
        img_filename = os.path.join(samples_dir, f'test_{iterations + 1}.jpg')
        img = imlib.immerge(np.concatenate([images_a, images_b] + G_images, axis=0), n_rows=8)
        imlib.imwrite(img, img_filename)


if __name__ == '__main__':
  main()
