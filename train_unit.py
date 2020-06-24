# Author: Mikita Sazanovich

import os

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
    x_aa, x_ba = tf.split(decoded_a, num_or_size_splits=2, axis=0)
    x_ab, x_bb = tf.split(decoded_b, num_or_size_splits=2, axis=0)
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
    self.gen_opt = optimization.create_optimizer_from_params(hyperparameters['optimizer'])
    self.dis_opt = optimization.create_optimizer_from_params(hyperparameters['optimizer'])
    self.control_opt = optimization.create_optimizer_from_params(hyperparameters['optimizer'])
    self.dis_loss_criterion = tf.keras.losses.BinaryCrossentropy()
    self.ll_loss_criterion_a = tf.keras.losses.MeanAbsoluteError()
    self.ll_loss_criterion_b = tf.keras.losses.MeanAbsoluteError()
    self.control_loss_fn = tf.keras.losses.MeanSquaredError()

  @tf.function
  def train_step(self, images_a, images_b, actions_a):
    D_loss_dict = self.dis_update(images_a, images_b)
    G_images, G_loss_dict = self.gen_update(images_a, images_b)
    C_loss_dict = self.control_update(images_a, actions_a)
    return D_loss_dict, G_images, G_loss_dict, C_loss_dict

  @tf.function
  def dis_update(self, images_a, images_b):
    training = True
    with tf.GradientTape() as t:
      x_aa, x_ba, x_ab, x_bb, shared = self.model.encode_ab_decode_aabb(images_a, images_b, training=training)
      data_a = tf.concat((images_a, x_ba), axis=0)
      data_b = tf.concat((images_b, x_ab), axis=0)
      res_a = self.model.dis_a(data_a, training=training)
      res_b = self.model.dis_b(data_b, training=training)
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

    dis_models = [self.model.dis_a, self.model.dis_b]
    self._compute_and_apply_gradients(dis_models, self.dis_opt, t, loss)

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
  def gen_update(self, images_a, images_b):
    training = True
    with tf.GradientTape() as t:
      x_aa, x_ba, x_ab, x_bb, shared = self.model.encode_ab_decode_aabb(images_a, images_b, training=training)
      x_bab, shared_bab = self.model.encode_a_decode_b(x_ba, training=training)
      x_aba, shared_aba = self.model.encode_b_decode_a(x_ab, training=training)
      outs_a = self.model.dis_a(x_ba, training=training)
      outs_b = self.model.dis_b(x_ab, training=training)
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

    gen_models = [
        self.model.encoder_a, self.model.encoder_b,
        self.model.encoder_shared, self.model.decoder_shared,
        self.model.decoder_a, self.model.decoder_b]
    self._compute_and_apply_gradients(gen_models, self.gen_opt, t, loss)

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

  @tf.function
  def control_update(self, images_a, actions_a):
    training = True
    with tf.GradientTape() as t:
      encoded_a = self.model.encoder_a(images_a, training=training)
      encoded_shared_a = self.model.encoder_shared(encoded_a, training=training)
      predictions_a = self.controller(encoded_shared_a, training=training)
      loss = self.control_loss_fn(actions_a, predictions_a)

    control_models = [self.model.encoder_a, self.model.encoder_shared, self.controller]
    self._compute_and_apply_gradients(control_models, self.control_opt, t, loss)

    C_loss_dict = {'loss': loss}
    return C_loss_dict

  @staticmethod
  def _compute_and_apply_gradients(models, optimizer, tape, loss) -> None:
    variables = []
    for model in models:
      variables.extend(model.trainable_variables)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))


def create_datasets(config, split):
  config_datasets = config['datasets']
  batch_size = config['hyperparameters']['batch_size']

  if split == 'train':
    domain_a_label = 'train_a'
    domain_b_label = 'train_b'
    training = True
  elif split == 'test':
    domain_a_label = 'test_a'
    domain_b_label = 'test_b'
    training = False
  else:
    raise ValueError(f'Split should be either train or test, got: {split}')

  image_a_dataset, image_a_length = data.create_image_dataset(config_datasets, domain_a_label, training)
  image_b_dataset, image_b_length = data.create_image_dataset(config_datasets, domain_b_label, training)
  action_a_dataset, action_a_length = data.create_action_dataset(config_datasets, domain_a_label)
  action_b_dataset, action_b_length = data.create_action_dataset(config_datasets, domain_b_label)

  # (image A, action A, image B, action B) dataset
  # TODO(niksaz): Zip function clips to the smallest length, meaning without repeat some samples are ignored.
  zip_dataset = tf.data.Dataset.zip((image_a_dataset, action_a_dataset, image_b_dataset, action_b_dataset))
  if training:
    shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048
    zip_dataset = zip_dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
    zip_dataset = zip_dataset.repeat(None)
    len_dataset = None
  else:
    len_dataset = (min(image_a_length, image_b_length) + batch_size - 1) // batch_size
  zip_dataset = zip_dataset.batch(batch_size, drop_remainder=False)
  return zip_dataset, len_dataset


def train_model(trainer, ab_train_dataset, config, checkpoint, samples_dir):
  optimizer_iterations = config['hyperparameters']['optimizer']['iterations']
  train_control_loss_mean = tf.keras.metrics.Mean()
  dataset_iter = iter(ab_train_dataset)
  for iterations in tqdm.tqdm(range(optimizer_iterations)):
    try:
      images_a, actions_a, images_b, actions_b = next(dataset_iter)
    except StopIteration:
      dataset_iter = iter(ab_train_dataset)
      images_a, actions_a, images_b, actions_b = next(dataset_iter)

    # Training ops
    D_loss_dict, G_images, G_loss_dict, C_loss_dict = trainer.train_step(images_a, images_b, actions_a)
    train_control_loss_mean.update_state(C_loss_dict['loss'])

    # Logging ops
    if (iterations + 1) % config['log_iterations'] == 0:
      C_loss_dict['loss'] = train_control_loss_mean.result()
      train_control_loss_mean.reset_states()
      tf2lib.summary(D_loss_dict, step=iterations, name='discriminator')
      tf2lib.summary(G_loss_dict, step=iterations, name='generator')
      tf2lib.summary(C_loss_dict, step=iterations, name='controller')
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
    if (iterations + 1) % config['checkpoint_save_iterations'] == 0 or iterations + 1 == optimizer_iterations:
      checkpoint.save(iterations + 1)


def test_model(unit_model, controller, ab_test_dataset, ab_test_length, samples_dir):
  training = False
  test_control_loss = tf.keras.metrics.MeanSquaredError()
  test_control_loss_mean = tf.keras.metrics.Mean()
  test_batches_to_save = 10
  test_dataset_iter = iter(ab_test_dataset)
  for iterations in tqdm.tqdm(range(ab_test_length)):
    try:
      images_a, actions_a, images_b, actions_b = next(test_dataset_iter)
    except StopIteration:
      break

    # Test ops
    x_aa, x_ba, x_ab, x_bb, shared = unit_model.encode_ab_decode_aabb(images_a, images_b, training=training)
    x_bab, _ = unit_model.encode_a_decode_b(x_ba, training=training)
    x_aba, _ = unit_model.encode_b_decode_a(x_ab, training=training)
    G_images = [x_aa, x_ba, x_ab, x_bb, x_aba, x_bab]
    shared_a, shared_b = tf.split(shared, num_or_size_splits=2, axis=0)
    predictions_a = controller(shared_a, training=training)
    control_loss = test_control_loss(actions_a, predictions_a)
    test_control_loss_mean.update_state(control_loss.numpy())

    # Displaying ops
    if (iterations + 1) % (ab_test_length // test_batches_to_save) == 0:
      img_filename = os.path.join(samples_dir, f'test_{iterations + 1}.jpg')
      img = imlib.immerge(np.concatenate([images_a, images_b] + G_images, axis=0), n_rows=8)
      imlib.imwrite(img, img_filename)

  loss_dict = {'loss_test_mean': test_control_loss_mean.result()}
  tf2lib.summary(loss_dict, step=0, name='controller')


def main():
  args = configuration.parse_args()
  config = configuration.load_config(args.config_path)
  print('args:', args)
  print('config:', config)

  utils.fix_random_seeds(config['hyperparameters']['seed'])

  ab_train_dataset, ab_train_length = create_datasets(config, split='train')
  ab_test_dataset, ab_test_length = create_datasets(config, split='test')

  unit_model = UNITModel(config)
  gen_hyperparameters = config['hyperparameters']['gen']
  z_ch = gen_hyperparameters['ch'] * 2 ** (gen_hyperparameters['n_enc_front_blk'] - 1)
  control_hyperparameters = config['hyperparameters']['control']
  controller = layers.Controller(z_ch, control_hyperparameters, 2)
  trainer = Trainer(unit_model, controller, config['hyperparameters'])

  samples_dir, summaries_dir, checkpoints_dir = utils.create_output_dirs(
      args.output_dir_base, 'unit', args.tag, ['samples', 'summaries', 'checkpoints'])

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

  if not args.skip_train:
    with summary_writer.as_default():
      train_model(trainer, ab_train_dataset, config, checkpoint, samples_dir)

  if not args.skip_test:
    # Restore from the test checkpoint
    test_checkpoint_dir = args.test_checkpoinst_dir if args.test_checkpoint_dir else checkpoints_dir
    test_latest_checkpoint = tf.train.latest_checkpoint(test_checkpoint_dir)
    print('The checkpoint for test is', test_latest_checkpoint)
    if not test_latest_checkpoint:
      raise ValueError('No checkpoint is found for the test stage. Specify it in --test_checkpoint_dir.')

    test_checkpoint = tf.train.Checkpoint(**checkpoint_dict)
    test_checkpoint.restore(test_latest_checkpoint).assert_existing_objects_matched()
    with summary_writer.as_default():
      test_model(unit_model, controller, ab_test_dataset, ab_test_length, samples_dir)

  if args.summarize:
    unit_model.encoder_a.model.summary()
    unit_model.encoder_b.model.summary()
    unit_model.encoder_shared.model.summary()
    unit_model.decoder_shared.model.summary()
    controller.model.summary()


if __name__ == '__main__':
  main()
