'''
This file implements the same as unet.py only using different format to build keras model.
This format can be cloned by tf.keras.models.clone_model, although we don't use this function.
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math

# data
DEFAULT_DTYPE=tf.float32
# architecture
Embedding_dims = 32
Attn_resolutions = [96, 128, 256, 512] #[16, 32]
block_depth = 2

def get_timestep_embedding(timesteps, embedding_dim: int):
    '''
    sinusoidal_embedding
    :param timesteps:
    :param embedding_dim:
    :return: sinusoidal embedding
    '''
    assert len(timesteps.shape) == 2 or 1  # and timesteps.dtype == tf.int32
    timesteps = tf.reshape(timesteps, [-1, ])

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=DEFAULT_DTYPE) * -emb)
    emb = tf.cast(timesteps, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    #output shape:[batch_size, embedding_dims]
    return emb

def SelfAttention(hidden_size):
    def apply(inputs):
        units = hidden_size
        norm = tf.keras.layers.LayerNormalization(axis=-1)
        query = tf.keras.layers.Dense(units)
        key = tf.keras.layers.Dense(units)
        value = tf.keras.layers.Dense(units)
        proj_d = tf.keras.layers.Dense(units)

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(units, DEFAULT_DTYPE) ** (-0.5)

        inputs = norm(inputs)
        q = query(inputs)
        k = key(inputs)
        v = value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = proj_d(proj)
        return inputs + proj
    return apply


def Resnet_block(channel_size):
    def apply(inputs, timeemb):
        d0 = tf.keras.layers.Dense(channel_size)
        c2d0 = tf.keras.layers.Conv2D(channel_size, kernel_size=1)
        c2d1 = tf.keras.layers.Conv2D(channel_size, kernel_size=3, padding="same", \
                                           activation=tf.keras.activations.swish)
        c2d2 = tf.keras.layers.Conv2D(channel_size, kernel_size=3, padding="same")
        bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        bn2 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        input_channel_size = inputs.shape[3]
        if input_channel_size == channel_size:
            residual = inputs
        else:
            residual = c2d0(inputs)
        x = bn1(inputs)
        x = c2d1(x)
        x = x + d0(timeemb)[:, None, None, :]
        x = tf.keras.activations.swish(bn2(x))
        x = tf.nn.dropout(x, rate=0.1)
        x = c2d2(x)
        x = tf.keras.layers.Add()([x, residual])
        return x
    return apply


def Down_block(channel_size, block_depth, attn_resolutions):
    def apply(inputs, skips, timeemb):
        rblock = []
        for _ in range(block_depth):
            rblock.append(Resnet_block(channel_size))
        attnblock = []
        for _ in range(block_depth):
            attnblock.append(SelfAttention(channel_size))

        x = inputs
        for i in range(block_depth):
            x = rblock[i](x, timeemb)
            if x.shape[1] in attn_resolutions:
                x = attnblock[i](x)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x, skips
    return apply


class Up_block(tf.keras.layers.Layer):
    def __init__(self, channel_size, block_depth, attn_resolutions):
        super(Up_block, self).__init__()
        self.channel_size = channel_size
        self.block_depth = block_depth
        self.attn_resolutions = attn_resolutions
        self.rblock=[]
        for _ in range(self.block_depth):
            self.rblock.append(Resnet_block(self.channel_size))
        self.attnblock=[]
        for _ in range(self.block_depth):
            self.attnblock.append(SelfAttention(self.channel_size))

    def call(self, inputs, skips, timeemb, training=None):
        self.timeemb = timeemb #[B,c*4]
        x = inputs
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for i in range(self.block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = self.rblock[i](x, self.timeemb, training)
            if x.shape[1] in self.attn_resolutions:
                x = self.attnblock[i](x, training)
        return x,skips

def Up_block(channel_size, block_depth, attn_resolutions):
    def apply(inputs, skips, timeemb):
        rblock = []
        for _ in range(block_depth):
            rblock.append(Resnet_block(channel_size))
        attnblock = []
        for _ in range(block_depth):
            attnblock.append(SelfAttention(channel_size))
        x = inputs
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for i in range(block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = rblock[i](x, timeemb)
            if x.shape[1] in attn_resolutions:
                x = attnblock[i](x)
        return x, skips
    return apply


#can be cloned in existing tf version
def get_network(image_size, widths, block_depth, embedding_dim=Embedding_dims, \
                 attn_resolutions=Attn_resolutions):
    #input
    noisy_images = tf.keras.Input(shape=(image_size, image_size, 3))
    t = tf.keras.Input(shape=(1, ))
    channel_size = widths

    #layer intialization
    input_dense = tf.keras.layers.Conv2D(channel_size[0], kernel_size=1)

    timeemb_dense0 = tf.keras.layers.Dense(embedding_dim * 4, activation=tf.keras.activations.swish)
    timeemb_dense1 = tf.keras.layers.Dense(embedding_dim * 4)

    downblocks = {}
    for nc in channel_size[:-1]:
        downblocks[nc] = Down_block(nc, block_depth, attn_resolutions)
    upblocks = {}
    for nc in channel_size[:-1]:
        upblocks[nc] = Up_block(nc, block_depth, attn_resolutions)

    midlocks_rb1 = []
    midlocks_rb2 = []
    midlocks_attnblock = SelfAttention(channel_size[-1])
    for _ in range(block_depth):
        midlocks_rb1.append(Resnet_block(channel_size[-1]))
        midlocks_rb2.append(Resnet_block(channel_size[-1]))

    output_bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
    output_con2d = tf.keras.layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")

    #call functions
    timeemb = get_timestep_embedding(t, embedding_dim)
    timeemb = timeemb_dense0(timeemb)  # [B, ch * 4]
    timeemb = timeemb_dense1(timeemb)  # [B, ch * 4]

    x = input_dense(noisy_images)

    skips = []
    for nc in channel_size[:-1]:
        #print(timeemb.shape)
        x, skips = downblocks[nc](x, skips, timeemb)

    for mb_i in midlocks_rb1:
        x = mb_i(x, timeemb)
    x = midlocks_attnblock(x)
    for mb_i in midlocks_rb2:
        x = mb_i(x, timeemb)

    for nc in reversed(channel_size[:-1]):
        x, skips = upblocks[nc](x, skips, timeemb)

    x = tf.keras.activations.swish(output_bn(x))
    x = output_con2d(x)

    return tf.keras.Model([noisy_images, t], x, name="attention_unet")