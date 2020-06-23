# Author: Mikita Sazanovich

import tensorflow as tf


# https://github.com/google-research/bert/blob/master/optimization.py
class BERTSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, optimizer_hyperparameters):
    super().__init__()
    self.num_train_steps = optimizer_hyperparameters['iterations']
    self.num_warmup_steps = optimizer_hyperparameters['warmup_iterations']
    self.init_lr = float(optimizer_hyperparameters['init_learning_rate'])

  def __call__(self, step):
    learning_rate = tf.constant(value=self.init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.compat.v1.train.polynomial_decay(
        learning_rate,
        step,
        self.num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if self.num_warmup_steps:
      step_int = tf.cast(step, tf.int32)
      warmup_steps_int = tf.constant(self.num_warmup_steps, dtype=tf.int32)

      step_float = tf.cast(step_int, tf.float32)
      warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

      warmup_percent_done = step_float / warmup_steps_float
      warmup_learning_rate = self.init_lr * warmup_percent_done

      is_warmup = tf.cast(step_int < warmup_steps_int, tf.float32)
      learning_rate = (
          (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    return learning_rate


# https://arxiv.org/pdf/1910.10683.pdf
class T5Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, optimizer_hyperparameters):
    super().__init__()
    self.num_warmup_steps = optimizer_hyperparameters['warmup_iterations']

  def __call__(self, step):
    warmup_steps_float = tf.constant(self.num_warmup_steps, dtype=tf.float32)
    step_or_warmup = tf.maximum(step, warmup_steps_float)
    lr = tf.math.rsqrt(step_or_warmup)
    return lr


class ConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, optimizer_hyperparameters):
    self.lr = float(optimizer_hyperparameters['lr'])

  def __call__(self, step):
    learning_rate = tf.constant(value=self.lr, shape=[], dtype=tf.float32)
    return learning_rate


def create_optimizer_from_params(optimizer_hyperparameters):
  lr_schedule_class = globals()[optimizer_hyperparameters['lr_schedule_class']]
  lr_schedule = lr_schedule_class(optimizer_hyperparameters)
  adam_params = {
      'learning_rate': lr_schedule,
      'beta_1': float(optimizer_hyperparameters['beta_1']),
      'beta_2': float(optimizer_hyperparameters['beta_2']),
      'epsilon': float(optimizer_hyperparameters['epsilon']),
  }
  if 'decay' in optimizer_hyperparameters:
    adam_params['decay'] = optimizer_hyperparameters['decay']
  optimizer = tf.keras.optimizers.Adam(**adam_params)
  return optimizer
