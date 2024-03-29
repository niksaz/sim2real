# Author: Mikita Sazanovich

import tensorflow as tf
import tensorflow_addons as tfa


def get_norm_layer(norm):
  if norm == 'none':
    return tf.keras.layers.Layer()
  elif norm == 'batch_norm':
    return tf.keras.layers.BatchNormalization()
  elif norm == 'instance_norm':
    return tfa.layers.InstanceNormalization()
  elif norm == 'layer_norm':
    return tf.keras.layers.LayerNormalization()


def Conv2DPadded(filters, kernel_size, strides, padding):
  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=[[0, 0], [padding, padding], [padding, padding], [0, 0]])


def ConvTranspose2d(filters, kernel_size, strides, padding, output_padding):
  return tf.keras.layers.Conv2DTranspose(
      filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding)


def Conv2DReLUBlock(input_filters, output_filters, kernel_size, strides, padding):
  layers = []
  layers.append(Conv2DPadded(output_filters, kernel_size, strides, padding))
  layers.append(tf.keras.layers.LeakyReLU(alpha=0.01))
  return tf.keras.Sequential(layers=layers)


def Conv2DTransposeReLUBlock(input_filters, output_filters, kernel_size, strides, padding, output_padding):
  layers = []
  layers.append(ConvTranspose2d(output_filters, kernel_size, strides, padding, output_padding))
  layers.append(tf.keras.layers.LeakyReLU(alpha=0.01))
  return tf.keras.Sequential(layers=layers)


def Conv3x3(inplanes, outplanes, strides=1):
  return Conv2DPadded(outplanes, kernel_size=3, strides=strides, padding=1)


def ResidualBlock(inplanes, planes, norm_layer, dropout=0.0):
  layers = []
  layers += [Conv3x3(inplanes, planes)]
  layers += [get_norm_layer(norm_layer)]
  layers += [tf.keras.layers.ReLU()]
  layers += [Conv3x3(planes, planes)]
  layers += [get_norm_layer(norm_layer)]
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
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)
    norm_layer = params['norm_layer']

    # Convolutional front-end
    layers = []
    layers += [Conv2DReLUBlock(input_dim, ch, kernel_size=7, strides=1, padding=3)]
    tch = ch
    for i in range(1, n_enc_front_blk):
      layers += [Conv2DReLUBlock(tch, tch * 2, kernel_size=3, strides=2, padding=1)]
      tch *= 2
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      layers += [ResidualBlock(tch, tch, norm_layer, dropout=res_dropout_ratio)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class EncoderShared(tf.keras.Model):
  def __init__(self, params):
    super(EncoderShared, self).__init__()
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)
    norm_layer = params['norm_layer']

    # Shared residual-blocks
    layers = []
    tch = ch * 2 ** (n_enc_front_blk - 1)
    for i in range(0, n_enc_shared_blk):
      layers += [ResidualBlock(tch, tch, norm_layer, dropout=res_dropout_ratio)]
    layers += [tf.keras.layers.GaussianNoise(stddev=1.0)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class DecoderShared(tf.keras.Model):
  def __init__(self, params):
    super(DecoderShared, self).__init__()
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)
    norm_layer = params['norm_layer']

    # Shared residual-blocks
    layers = []
    tch = ch * 2 ** (n_enc_front_blk - 1)
    for i in range(0, n_dec_shared_blk):
      layers += [ResidualBlock(tch, tch, norm_layer, dropout=res_dropout_ratio)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class Decoder(tf.keras.Model):
  def __init__(self, params):
    super(Decoder, self).__init__()
    input_dim = params['input_dim']
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_dec_res_blk = params['n_dec_res_blk']
    n_dec_front_blk = params['n_dec_front_blk']
    res_dropout_ratio = params.get('res_dropout_ratio', 0.0)
    norm_layer = params['norm_layer']

    # Residual-block front-end
    layers = []
    tch = ch * 2 ** (n_enc_front_blk - 1)
    for i in range(0, n_dec_res_blk):
      layers += [ResidualBlock(tch, tch, norm_layer, dropout=res_dropout_ratio)]
    # Convolutional back-end
    for i in range(0, n_dec_front_blk - 1):
      layers += [Conv2DTransposeReLUBlock(tch, tch // 2, kernel_size=3, strides=2, padding='same', output_padding=1)]
      tch = tch // 2
    layers += [ConvTranspose2d(filters=input_dim, kernel_size=1, strides=1, padding='same', output_padding=0)]
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
    norm_layer = params['norm_layer']

    layers = []
    layers += [Conv2DReLUBlock(input_dim, ch, kernel_size=3, strides=2, padding=1)]
    tch = ch
    for i in range(1, n_layer):
      layers += [Conv2DReLUBlock(tch, tch * 2, kernel_size=3, strides=2, padding=1)]
      tch *= 2
    layers += [tf.keras.layers.Conv2D(1, kernel_size=1, strides=1)]
    layers.append(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
    self.model = tf.keras.Sequential(layers=layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class Downstreamer(tf.keras.Model):
  def __init__(self, params):
    super(Downstreamer, self).__init__()
    layers = []
    layers.append(tf.keras.layers.GlobalMaxPool2D())  # B x C
    for fc_layer in params['fc_layers']:
      # https://github.com/google-research/google-research/blob/084c18934c353207662aba0db6db52850029faf2/tcc/models.py#L50
      # layers.append(get_norm_layer('batch_norm'))  # B x FC
      # layers.append(tf.keras.layers.Dense(fc_layer))  # B x FC
      # layers.append(tf.keras.layers.ReLU())  # B x FC
      # AIDO3
      layers.append(get_norm_layer('batch_norm'))
      layers.append(tf.keras.layers.LeakyReLU(alpha=0.01))
      layers.append(tf.keras.layers.Dense(fc_layer))
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class Controller(tf.keras.Model):
  def __init__(self, input_dim, control_hyperparameters, output_dim):
    super(Controller, self).__init__()
    layers = []
    fc_layers_with_output = control_hyperparameters['fc_layers'] + [output_dim]
    for fc_layer_index, fc_layer in enumerate(fc_layers_with_output):
      # AIDO3
      # TODO: Think about the order (should be the same in Downstreamer)>
      layers.append(get_norm_layer('batch_norm'))
      layers.append(tf.keras.layers.LeakyReLU(alpha=0.01))
      layers.append(tf.keras.layers.Dropout(rate=0.1))
      layers.append(tf.keras.layers.Dense(fc_layer))
    layers.append(tf.keras.layers.Activation('tanh'))
    self.model = tf.keras.Sequential(layers=layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class T46Controller(tf.keras.Model):
  def __init__(self, control_hyperparameters, output_dim):
    super(T46Controller, self).__init__()
    layers = []
    for fc_layer in control_hyperparameters['fc_layers']:
      layers.append(get_norm_layer('batch_norm'))
      layers.append(tf.keras.layers.Dense(fc_layer))
      layers.append(tf.keras.layers.ReLU())
      layers.append(tf.keras.layers.Dropout(rate=0.1))
    layers.append(tf.keras.layers.Dense(output_dim))
    layers.append(tf.keras.layers.Activation('tanh'))
    self.model = tf.keras.Sequential(layers=layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


def T46DiscriminatorConv2dBlock(n_out, kernel_size, stride, padding):
  layers = [
    Conv2DPadded(n_out, kernel_size=kernel_size, strides=1, padding=padding),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(pool_size=stride),
  ]
  return tf.keras.Sequential(layers=layers)


class T46DiscriminatorHead(tf.keras.Model):
  def __init__(self, params):
    super(T46DiscriminatorHead, self).__init__()
    ch = params['ch']
    layers = []
    layers.append(T46DiscriminatorConv2dBlock(ch, kernel_size=5, stride=2, padding=2))
    layers.append(tf.keras.layers.Dropout(0.1))
    self.model = tf.keras.Sequential(layers=layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class T46DiscriminatorShared(tf.keras.Model):
  def __init__(self, params):
    super(T46DiscriminatorShared, self).__init__()
    ch = params['ch']
    layers = []
    layers.append(T46DiscriminatorConv2dBlock(ch * 2, kernel_size=5, stride=2, padding=2))
    layers.append(tf.keras.layers.Dropout(0.3))
    layers.append(T46DiscriminatorConv2dBlock(ch * 4, kernel_size=5, stride=2, padding=2))
    layers.append(tf.keras.layers.Dropout(0.5))
    layers.append(T46DiscriminatorConv2dBlock(ch * 8, kernel_size=5, stride=2, padding=2))
    layers.append(tf.keras.layers.Dropout(0.5))
    layers.append(Conv2DPadded(1, kernel_size=(2, 4), strides=1, padding=0))
    layers.append(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
    self.model = tf.keras.Sequential(layers=layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


def T46LeakyReLUBNNSConv2d(
    input_filters, output_filters, kernel_size, strides, padding, norm_layer):
  layers = []
  layers.append(Conv2DPadded(output_filters, kernel_size, strides, padding))
  layers.append(get_norm_layer(norm_layer))
  layers.append(tf.keras.layers.LeakyReLU(alpha=0.01))
  return tf.keras.Sequential(layers=layers)


def T46LeakyReLUBNNSConvTranspose2d(
    input_filters, output_filters, kernel_size, strides, padding, output_padding, norm_layer):
  layers = []
  layers.append(ConvTranspose2d(output_filters, kernel_size, strides, padding, output_padding))
  layers.append(get_norm_layer(norm_layer))
  layers.append(tf.keras.layers.LeakyReLU(alpha=0.01))
  return tf.keras.Sequential(layers=layers)


class T46Encoder(tf.keras.Model):
  def __init__(self, params):
    super(T46Encoder, self).__init__()
    input_dim = params['input_dim']
    ch = params['ch']
    norm_layer = params['norm_layer']

    # Convolutional front-end
    layers = []
    layers += [T46LeakyReLUBNNSConv2d(input_dim, ch, kernel_size=(5, 5), strides=2, padding=2, norm_layer=norm_layer)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class T46EncoderShared(tf.keras.Model):
  def __init__(self, params):
    super(T46EncoderShared, self).__init__()
    ch = params['ch']
    norm_layer = params['norm_layer']

    layers = []
    layers += [T46LeakyReLUBNNSConv2d(ch * 1, ch * 2, kernel_size=(5, 5), strides=2, padding=2, norm_layer=norm_layer)]
    layers += [T46LeakyReLUBNNSConv2d(ch * 2, ch * 4, kernel_size=(8, 16), strides=1, padding=0, norm_layer=norm_layer)]
    layers += [T46LeakyReLUBNNSConv2d(ch * 4, ch * 8, kernel_size=(1, 1), strides=1, padding=0, norm_layer=norm_layer)]
    layers += [tf.keras.layers.GaussianNoise(stddev=1.0)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    result = self.model(inputs, **kwargs)
    return result


class T46DecoderShared(tf.keras.Model):
  def __init__(self, params):
    super(T46DecoderShared, self).__init__()
    ch = params['ch']
    norm_layer = params['norm_layer']

    layers = []
    layers += [T46LeakyReLUBNNSConvTranspose2d(
        ch * 8, ch * 8, kernel_size=(4, 8), strides=2, padding='valid', output_padding=0, norm_layer=norm_layer)]
    layers += [T46LeakyReLUBNNSConvTranspose2d(
        ch * 8, ch * 4, kernel_size=(3, 3), strides=2, padding='same', output_padding=1, norm_layer=norm_layer)]
    layers += [T46LeakyReLUBNNSConvTranspose2d(
        ch * 4, ch * 2, kernel_size=(3, 3), strides=2, padding='same', output_padding=1, norm_layer=norm_layer)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)


class T46Decoder(tf.keras.Model):
  def __init__(self, params):
    super(T46Decoder, self).__init__()
    input_dim = params['input_dim']
    ch = params['ch']
    norm_layer = params['norm_layer']

    layers = []
    layers += [T46LeakyReLUBNNSConvTranspose2d(
        ch * 2, ch * 1, kernel_size=(3, 3), strides=2, padding='same', output_padding=1, norm_layer=norm_layer)]
    layers += [ConvTranspose2d(input_dim, kernel_size=1, strides=1, padding='same', output_padding=0)]
    layers += [tf.keras.layers.Activation(tf.keras.activations.tanh)]
    self.model = tf.keras.Sequential(layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)
