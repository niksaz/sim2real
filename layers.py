# Author: Mikita Sazanovich

import tensorflow as tf
import tensorflow_addons as tfa


def get_norm_layer(norm):
  if norm == 'none':
    return lambda x: x
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


def Conv2DTransposePaddedSame(filters, kernel_size, strides, padding, output_padding):
  if padding != kernel_size // 2:
    raise NotImplementedError(
        f'padding {padding} and kernel_size {kernel_size} are not supported by Conv2DTranspose.')
  assert kernel_size // 2 == padding
  return tf.keras.layers.Conv2DTranspose(
          filters=filters, kernel_size=kernel_size, strides=strides, padding='same', output_padding=output_padding)


def NormReLUConv2DBlock(
    input_filters, output_filters, kernel_size, strides, padding, norm_layer):
  layers = []
  layers.append(Conv2DPadded(output_filters, kernel_size, strides, padding))
  # layers.append(tf.keras.layers.LeakyReLU())  # TODO(sazanovich): alpha = 0.3 while in torch it is 0.01
  # layers.append(get_norm_layer(norm_layer))
  layers.append(tf.keras.layers.ReLU())
  return tf.keras.Sequential(layers=layers)


def NormReLUConv2DTransposeBlock(
    input_filters, output_filters, kernel_size, strides, padding, output_padding, norm_layer):
  layers = []
  layers.append(Conv2DTransposePaddedSame(output_filters, kernel_size, strides, padding, output_padding))
  # layers.append(tf.keras.layers.LeakyReLU())  # TODO(sazanovich): alpha = 0.3 while in torch it is 0.01
  # layers.append(get_norm_layer(norm_layer))
  layers.append(tf.keras.layers.ReLU())
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
  # layers += [tf.keras.layers.ReLU()]
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
    norm_layer = params['norm_layer']

    # Convolutional front-end
    layers = []
    layers += [NormReLUConv2DBlock(input_dim, ch, kernel_size=7, strides=1, padding=3, norm_layer=norm_layer)]
    tch = ch
    for i in range(1, n_enc_front_blk):
      layers += [NormReLUConv2DBlock(tch, tch * 2, kernel_size=3, strides=2, padding=1, norm_layer=norm_layer)]
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
    input_dim = params['input_dim']
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_enc_res_blk = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
    n_dec_res_blk = params['n_dec_res_blk']
    n_dec_front_blk = params['n_dec_front_blk']
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
    input_dim = params['input_dim']
    ch = params['ch']
    n_enc_front_blk = params['n_enc_front_blk']
    n_enc_res_blk = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
    n_dec_res_blk = params['n_dec_res_blk']
    n_dec_front_blk = params['n_dec_front_blk']
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
    n_enc_res_blk = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_dec_shared_blk = params['n_dec_shared_blk']
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
      layers += [NormReLUConv2DTransposeBlock(
          tch, tch // 2, kernel_size=3, strides=2, padding=1, output_padding=1, norm_layer=norm_layer)]
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
    norm_layer = params['norm_layer']

    model = []
    model += [NormReLUConv2DBlock(input_dim, ch, kernel_size=3, strides=2, padding=1, norm_layer=norm_layer)]
    tch = ch
    for i in range(1, n_layer):
      model += [NormReLUConv2DBlock(tch, tch * 2, kernel_size=3, strides=2, padding=1, norm_layer=norm_layer)]
      tch *= 2
    model += [tf.keras.layers.Conv2D(1, kernel_size=1, strides=1)]
    self.model = tf.keras.Sequential(layers=model)

  def __call__(self, inputs, **kwargs):
    out = self.model(inputs, **kwargs)
    out = tf.reshape(out, [-1])
    return [out]


class Controller(tf.keras.Model):
  def __init__(self, input_dim, control_hyperparameters, output_dim):
    super(Controller, self).__init__()
    # B x 8 x 16 x 32 -> 4096 -> 16 -> 2 | n_layer=0
    # B x 4 x 8 x 32 -> 1024 -> 16 -> 2 | n_layer=1
    # B x 2 x 4 x 32 -> 256 -> 16 -> 2 | n_layer=2
    # B x 1 x 2 x 32 -> 64 -> 16 -> 2 | n_layer=3
    layers = []
    for layer in range(control_hyperparameters['n_layer']):
      conv_block = NormReLUConv2DBlock(
          input_dim, input_dim, kernel_size=3, strides=2, padding=1, norm_layer='batch_norm')
      layers.append(conv_block)
    layers.append(tf.keras.layers.Flatten())
    fc_layers_with_output = control_hyperparameters['fc_layers'] + [output_dim]
    for fc_layer in fc_layers_with_output:
      layers.append(get_norm_layer('batch_norm'))
      layers.append(tf.keras.layers.ReLU())
      layers.append(tf.keras.layers.Dense(fc_layer))
    self.model = tf.keras.Sequential(layers=layers)

  def __call__(self, inputs, **kwargs):
    return self.model(inputs, **kwargs)
