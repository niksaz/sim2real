# Author: Mikita Sazanovich

import unittest

import tensorflow as tf

import utils


class UtilsTest(unittest.TestCase):
  def test_mse_loss(self):
    loss = utils.get_loss_fn('mse')
    a = tf.constant([[0.0, 0.0], [0.0, 0.0]])
    b = tf.constant([[1.0, 1.0], [2.0, 2.0]])
    loss_result = loss(a, b)
    # Should be 1/2 * (1/2 * (1**2 + 1**2) + 1/2 * (2**2 + 2**2)) = 2.5.
    self.assertEqual(tf.constant(2.5), loss_result)

  def test_mae_loss(self):
    loss = utils.get_loss_fn('mae')
    a = tf.constant([[0.0, 0.0], [0.0, 0.0]])
    b = tf.constant([[1.0, 1.0], [2.0, 2.0]])
    loss_result = loss(a, b)
    # Should be 1/2 * (1/2 * (1 + 1) + 1/2 * (2 + 2)) = 1.5.
    self.assertEqual(tf.constant(1.5), loss_result)

  def test_kl_computation(self):
    z = tf.constant([[1.0, 2.0, 3.0]])
    kl_result = utils.compute_kl(z)
    self.assertEqual(tf.constant(14 / 3), kl_result)

  def test_kl_zero(self):
    z = tf.constant([[0.0, 0.0, 0.0]])
    kl_result = utils.compute_kl(z)
    self.assertEqual(tf.constant(0.0), kl_result)
