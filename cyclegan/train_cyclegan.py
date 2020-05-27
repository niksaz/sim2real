# Author: Mikita Sazanovich

import os
import argparse
import functools

import tensorflow as tf
import numpy as np
import pylib
import imlib
import tf2lib
import tf2gan
import tqdm

import data
from cyclegan import module


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='horse2zebra')
  parser.add_argument('--datasets_dir', default='datasets')
  parser.add_argument('--load_size_width', type=int, default=286)  # loaded images are resized to this width
  parser.add_argument('--load_size_height', type=int, default=286)  # and this height
  parser.add_argument('--crop_size_width', type=int, default=256)  # then cropped to this width
  parser.add_argument('--crop_size_height', type=int, default=256)  # and this height
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
  parser.add_argument('--lr', type=float, default=0.0002)
  parser.add_argument('--beta_1', type=float, default=0.5)
  parser.add_argument('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
  parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
  parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
  parser.add_argument('--cycle_loss_weight', type=float, default=10.0)
  parser.add_argument('--identity_loss_weight', type=float, default=0.0)
  parser.add_argument('--pool_size', type=int, default=50)  # pool size to store fake samples
  args = parser.parse_args()
  return args


def train():
  # ===================================== Args =====================================
  args = parse_args()
  output_dir = os.path.join('output', args.dataset)
  os.makedirs(output_dir, exist_ok=True)
  settings_path = os.path.join(output_dir, 'settings.json')
  pylib.args_to_json(settings_path, args)

  # ===================================== Data =====================================
  A_img_paths = pylib.glob(os.path.join(args.datasets_dir, args.dataset, 'trainA'), '*.png')
  B_img_paths = pylib.glob(os.path.join(args.datasets_dir, args.dataset, 'trainB'), '*.png')
  print(f'len(A_img_paths) = {len(A_img_paths)}')
  print(f'len(B_img_paths) = {len(B_img_paths)}')
  load_size = [args.load_size_height, args.load_size_width]
  crop_size = [args.crop_size_height, args.crop_size_width]
  A_B_dataset, len_dataset = data.make_zip_dataset(
    A_img_paths, B_img_paths, args.batch_size, load_size, crop_size, training=True, repeat=False)

  A2B_pool = data.ItemPool(args.pool_size)
  B2A_pool = data.ItemPool(args.pool_size)

  A_img_paths_test = pylib.glob(os.path.join(args.datasets_dir, args.dataset, 'testA'), '*.png')
  B_img_paths_test = pylib.glob(os.path.join(args.datasets_dir, args.dataset, 'testB'), '*.png')
  A_B_dataset_test, _ = data.make_zip_dataset(
    A_img_paths_test, B_img_paths_test, args.batch_size, load_size, crop_size, training=False, repeat=True)

  # ===================================== Models =====================================
  model_input_shape = crop_size + [3]  # [args.crop_size_height, args.crop_size_width, 3]

  G_A2B = module.ResnetGenerator(input_shape=model_input_shape, n_blocks=6)
  G_B2A = module.ResnetGenerator(input_shape=model_input_shape, n_blocks=6)

  D_A = module.ConvDiscriminator(input_shape=model_input_shape)
  D_B = module.ConvDiscriminator(input_shape=model_input_shape)

  d_loss_fn, g_loss_fn = tf2gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
  cycle_loss_fn = tf.losses.MeanAbsoluteError()
  identity_loss_fn = tf.losses.MeanAbsoluteError()

  G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
  D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
  G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
  D_optimizer = tf.keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)

  # ===================================== Training steps =====================================
  @tf.function
  def train_generators(A, B):
    with tf.GradientTape() as t:
      A2B = G_A2B(A, training=True)
      B2A = G_B2A(B, training=True)
      A2B2A = G_B2A(A2B, training=True)
      B2A2B = G_A2B(B2A, training=True)
      A2A = G_B2A(A, training=True)
      B2B = G_A2B(B, training=True)

      A2B_d_logits = D_B(A2B, training=True)
      B2A_d_logits = D_A(B2A, training=True)

      A2B_g_loss = g_loss_fn(A2B_d_logits)
      B2A_g_loss = g_loss_fn(B2A_d_logits)
      A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
      B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
      A2A_id_loss = identity_loss_fn(A, A2A)
      B2B_id_loss = identity_loss_fn(B, B2B)

      G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (
            A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}

  @tf.function
  def train_discriminators(A, B, A2B, B2A):
    with tf.GradientTape() as t:
      A_d_logits = D_A(A, training=True)
      B2A_d_logits = D_A(B2A, training=True)
      B_d_logits = D_B(B, training=True)
      A2B_d_logits = D_B(A2B, training=True)

      A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
      B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
      D_A_gp = tf2gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
      D_B_gp = tf2gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

      D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}

  def train_step(A, B):
    A2B, B2A, G_loss_dict = train_generators(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_discriminators(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict

  @tf.function
  def sample(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B

  # ===================================== Runner code =====================================
  # epoch counter
  ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

  # checkpoint
  checkpoint = tf2lib.Checkpoint(
    dict(G_A2B=G_A2B, G_B2A=G_B2A, D_A=D_A, D_B=D_B, G_optimizer=G_optimizer, D_optimizer=D_optimizer, ep_cnt=ep_cnt),
    os.path.join(output_dir, 'checkpoints'),
    max_to_keep=5)
  try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
  except Exception as e:
    print(e)

  # summary
  train_summary_writer = tf.summary.create_file_writer(os.path.join(output_dir, 'summaries', 'train'))

  # sample
  test_iter = iter(A_B_dataset_test)
  sample_dir = os.path.join(output_dir, 'samples_training')
  os.makedirs(sample_dir, exist_ok=True)

  # main loop
  with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
      if ep < ep_cnt:
        continue

      # update epoch counter
      ep_cnt.assign_add(1)

      # train for an epoch
      for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
        G_loss_dict, D_loss_dict = train_step(A, B)

        # # summary
        tf2lib.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
        tf2lib.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
        tf2lib.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                       name='learning rate')

        # sample
        if G_optimizer.iterations.numpy() % 100 == 0:
          A, B = next(test_iter)
          A2B, B2A, A2B2A, B2A2B = sample(A, B)
          img = imlib.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=6)
          imlib.imwrite(img, os.path.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

      # save checkpoint
      checkpoint.save(ep)


if __name__ == '__main__':
  train()
