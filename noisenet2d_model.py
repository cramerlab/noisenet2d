
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def leaky_relu_norm(inputs, is_training):

  inputs = tf.layers.batch_normalization(inputs=inputs, 
                                         axis=1,
                                         momentum=_BATCH_NORM_DECAY, 
                                         epsilon=_BATCH_NORM_EPSILON, 
                                         center=True, 
                                         scale=True, 
                                         training=is_training, 
                                         fused=True)

  inputs = tf.nn.leaky_relu(inputs, alpha=0.1)
  return inputs

def leaky_relu(inputs):

  inputs = tf.nn.leaky_relu(inputs, alpha=0.1)
  return inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, dilation):

  return tf.layers.conv2d(inputs=inputs, 
                          filters=filters, 
                          kernel_size=kernel_size, 
                          strides=strides,
                          dilation_rate=dilation,
                          padding='SAME', 
                          use_bias=False,
                          data_format='channels_first',
                          kernel_initializer=tf.variance_scaling_initializer())


def deconv2d_fixed_padding(inputs, filters, kernel_size, strides):

  return tf.layers.conv2d_transpose(inputs=inputs, 
                                    filters=filters, 
                                    kernel_size=kernel_size, 
                                    strides=strides,
                                    padding='SAME', 
                                    use_bias=False,
                                    data_format='channels_first',
                                    kernel_initializer=tf.variance_scaling_initializer())


def maxpool2d(inputs):

  return tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME', data_format='channels_first')



def noisenet2d_generator():

  def model(inputs, is_training):
  
    training_batchnorm = True
    concat_dim = 1
    
    channels_encode = 32
    channels_decode = 64
  
    # input is channels_last, but we need channels_first
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    
    encode0 = leaky_relu_norm(conv2d_fixed_padding(inputs, channels_encode, 3, 1, 1), is_training)               # 256
    encode1 = leaky_relu_norm(conv2d_fixed_padding(encode0, channels_encode, 3, 1, 1), is_training)              # 256
    pool1 = maxpool2d(encode1)    # 128
    
    encode2 = leaky_relu_norm(conv2d_fixed_padding(pool1, channels_encode, 3, 1, 1), is_training)              # 128
    pool2 = maxpool2d(encode2)    # 64
    
    encode3 = leaky_relu_norm(conv2d_fixed_padding(pool2, channels_encode, 3, 1, 1), is_training)              # 64
    pool3 = maxpool2d(encode3)    # 32
    
    encode4 = leaky_relu_norm(conv2d_fixed_padding(pool3, channels_encode, 3, 1, 1), is_training)              # 32
    pool4 = maxpool2d(encode4)    # 16
    
    encode5 = leaky_relu_norm(conv2d_fixed_padding(pool4, channels_encode, 3, 1, 1), is_training)              # 16
    pool5 = maxpool2d(encode5)    # 8
    
    encode6 = leaky_relu_norm(conv2d_fixed_padding(pool5, channels_encode, 3, 1, 1), is_training)              # 8
    
    deconv5 = deconv2d_fixed_padding(encode6, channels_encode, 3, 2)    # 16
    concat5 = tf.concat([deconv5, pool4], concat_dim)                  # 16
    decode5a = leaky_relu_norm(conv2d_fixed_padding(concat5, channels_decode, 3, 1, 1), is_training)     # 16
    decode5b = leaky_relu_norm(conv2d_fixed_padding(decode5a, channels_decode, 3, 1, 1), is_training)    # 16
    
    deconv4 = deconv2d_fixed_padding(decode5b, channels_decode, 3, 2)    # 32
    concat4 = tf.concat([deconv4, pool3], concat_dim)                  # 32
    decode4a = leaky_relu_norm(conv2d_fixed_padding(concat4, channels_decode, 3, 1, 1), is_training)     # 32
    decode4b = leaky_relu_norm(conv2d_fixed_padding(decode4a, channels_decode, 3, 1, 1), is_training)    # 32
    
    deconv3 = deconv2d_fixed_padding(decode4b, channels_decode, 3, 2)    # 64
    concat3 = tf.concat([deconv3, pool2], concat_dim)                  # 64
    decode3a = leaky_relu_norm(conv2d_fixed_padding(concat3, channels_decode, 3, 1, 1), is_training)     # 64
    decode3b = leaky_relu_norm(conv2d_fixed_padding(decode3a, channels_decode, 3, 1, 1), is_training)    # 64
    
    deconv2 = deconv2d_fixed_padding(decode3b, channels_decode, 3, 2)    # 128
    concat2 = tf.concat([deconv2, pool1], concat_dim)                  # 128
    decode2a = leaky_relu_norm(conv2d_fixed_padding(concat2, channels_decode, 3, 1, 1), is_training)     # 128
    decode2b = leaky_relu_norm(conv2d_fixed_padding(decode2a, channels_decode, 3, 1, 1), is_training)    # 128
    
    deconv1 = deconv2d_fixed_padding(decode2b, channels_decode, 3, 2)    # 256
    concat1 = tf.concat([deconv1, inputs], concat_dim)                  # 256
    decode1a = leaky_relu(conv2d_fixed_padding(concat1, 48, 3, 1, 1))     # 256
    decode1b = leaky_relu(conv2d_fixed_padding(decode1a, 24, 3, 1, 1))    # 256
    
    decode1c = conv2d_fixed_padding(decode1b, 1, 3, 1, 1)    # 256
    
    # and back to channels_last
    inputs = tf.transpose(decode1c, [0, 2, 3, 1])
    #inputs = decode1c
    
    return inputs

  return model