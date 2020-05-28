# Author: Mikita Sazanovich

import tensorflow as tf
import tensorflow_addons as tfa


def Conv2DPadded(filters, kernel_size, strides, padding):
  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=[[0, 0], [padding, padding], [padding, padding], [0, 0]])


def Conv2DTransposePaddedSame(filters, kernel_size, strides, padding, output_padding):
  if padding != kernel_size // 2:
    raise NotImplementedError(
        f'padding {padding} and kernel_size {kernel_size} are not supported by Conv2DTranspose.')
  assert kernel_size // 2 == padding
  return tf.keras.layers.Conv2DTranspose(
          filters=filters, kernel_size=kernel_size, strides=strides, padding='same', output_padding=output_padding)


def LeakyReLUConv2D(input_filters, output_filters, kernel_size, strides, padding):
  layers = []
  layers.append(Conv2DPadded(output_filters, kernel_size, strides, padding))
  layers.append(tf.keras.layers.LeakyReLU())  # TODO(sazanovich): alpha = 0.3 while in torch it is 0.01
  return tf.keras.Sequential(layers=layers)


def LeakyReLUConv2DTranspose(input_filters, output_filters, kernel_size, strides, padding, output_padding):
  layers = []
  layers.append(Conv2DTransposePaddedSame(output_filters, kernel_size, strides, padding, output_padding))
  layers.append(tf.keras.layers.LeakyReLU())  # TODO(sazanovich): alpha = 0.3 while in torch it is 0.01
  return tf.keras.Sequential(layers=layers)


def Conv3x3(inplanes, outplanes, strides=1):
  return Conv2DPadded(outplanes, kernel_size=3, strides=strides, padding=1)


def INSResBlock(inplanes, planes, strides=1, dropout=0.0):
  layers = []
  layers += [Conv3x3(inplanes, planes, strides)]
  layers += [tfa.layers.InstanceNormalization()]
  layers += [tf.keras.layers.ReLU()]
  layers += [Conv3x3(planes, planes)]
  layers += [tfa.layers.InstanceNormalization()]
  if dropout > 0:
    layers += [tf.keras.layers.Dropout(rate=dropout)]
  block = tf.keras.Sequential(layers=layers)

  input_shape = (None, None, inplanes)
  inputs = tf.keras.Input(shape=input_shape)
  outputs = inputs + block(inputs)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  return model


class Encoder(tf.keras.Model):
  def __init__(self, params):
    super(Encoder, self).__init__()
    input_dim = params['input_dim']
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_enc_res_blk = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
    n_dec_res_blk = params['n_dec_res_blk']
    n_dec_front_blk = params['n_dec_front_blk']
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)

    # Convolutional front-end
    layers = []
    layers += [LeakyReLUConv2D(input_dim, ch, kernel_size=7, strides=1, padding=3)]
    tch = ch
    for i in range(1, n_enc_front_blk):
      layers += [LeakyReLUConv2D(tch, tch * 2, kernel_size=3, strides=2, padding=1)]
      tch *= 2
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      layers += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class EncoderShared(tf.keras.Model):
  def __init__(self, params):
    super(EncoderShared, self).__init__()
    input_dim = params['input_dim']
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_enc_res_blk = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
    n_dec_res_blk = params['n_dec_res_blk']
    n_dec_front_blk = params['n_dec_front_blk']
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)

    # Shared residual-blocks
    layers = []
    tch = ch * 2 ** (n_enc_front_blk - 1)
    for i in range(0, n_enc_shared_blk):
      layers += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    layers += [tf.keras.layers.GaussianNoise(stddev=1.0)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class DecoderShared(tf.keras.Model):
  def __init__(self, params):
    super(DecoderShared, self).__init__()
    input_dim = params['input_dim']
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_enc_res_blk = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
    n_dec_res_blk = params['n_dec_res_blk']
    n_dec_front_blk = params['n_dec_front_blk']
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)

    # Shared residual-blocks
    layers = []
    tch = ch * 2 ** (n_enc_front_blk - 1)
    for i in range(0, n_dec_shared_blk):
      layers += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class Decoder(tf.keras.Model):
  def __init__(self, params):
    super(Decoder, self).__init__()
    input_dim = params['input_dim']
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_enc_res_blk = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
    n_dec_res_blk = params['n_dec_res_blk']
    n_dec_front_blk = params['n_dec_front_blk']
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)

    # Residual-block front-end
    layers = []
    tch = ch * 2 ** (n_enc_front_blk - 1)
    for i in range(0, n_dec_res_blk):
      layers += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # Convolutional back-end
    for i in range(0, n_dec_front_blk - 1):
      layers += [LeakyReLUConv2DTranspose(tch, tch // 2, kernel_size=3, strides=2, padding=1, output_padding=1)]
      tch = tch // 2
    layers += [Conv2DTransposePaddedSame(filters=input_dim, kernel_size=1, strides=1, padding=0, output_padding=0)]
    layers += [tf.keras.layers.Activation(tf.keras.activations.tanh)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class Discriminator(tf.keras.Model):
  def __init__(self, params):
    super(Discriminator, self).__init__()
    ch = params['ch']
    input_dim = params['input_dim']
    n_layer = params['n_layer']

    model = []
    model += [LeakyReLUConv2D(input_dim, ch, kernel_size=3, strides=2, padding=1)]
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2D(tch, tch * 2, kernel_size=3, strides=2, padding=1)]
      tch *= 2
    model += [tf.keras.layers.Conv2D(1, kernel_size=1, strides=1)]
    self.model = tf.keras.Sequential(layers=model)

  def __call__(self, inputs, **kwargs):
    out = self.model(inputs, **kwargs)
    out = tf.reshape(out, [-1])
    return [out]


class Controller(tf.keras.Model):
  def __init__(self, params, ch):
    super(Controller, self).__init__()
    n_layer = params['n_layer']

    layers = []
    # tch = ch
    # for i in range(n_layer):
    #   layers += [LeakyReLUConv2D(tch, tch * 2, kernel_size=3, strides=2, padding=1)]
    #   tch *= 2
    layers.append(tf.keras.layers.AveragePooling2D(pool_size=(8, 16)))
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(16))
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(tf.keras.layers.Dense(2))
    self.sequential = tf.keras.Sequential(layers=layers)

  def __call__(self, inputs, **kwargs):
    return self.sequential(inputs, **kwargs)
