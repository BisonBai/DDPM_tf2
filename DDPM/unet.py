'''
This file implements the same as unet_clonable.py only using different format to build keras model.
This format can be cloned by tf.keras.models.clone_model, although we don't use this function.
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math

# data
DEFAULT_DTYPE=tf.float32
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20
# architecture
Embedding_dims = 32
Attn_resolutions = [16]
block_depth = 2
# optimization
batch_size = 64

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


class SelfAttention(tf.keras.layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
        self.norm = tf.keras.layers.LayerNormalization(axis = -1)
        self.query = tf.keras.layers.Dense(units)
        self.key = tf.keras.layers.Dense(units)
        self.value = tf.keras.layers.Dense(units)
        self.proj = tf.keras.layers.Dense(units)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, DEFAULT_DTYPE) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj



class Resnet_block(tf.keras.layers.Layer):
    def __init__(self, channel_size):
        super(Resnet_block, self).__init__()
        self.channel_size = channel_size
        self.d0 = tf.keras.layers.Dense(self.channel_size)
        self.c2d0 = tf.keras.layers.Conv2D(self.channel_size, kernel_size=1)
        self.c2d1 = tf.keras.layers.Conv2D(self.channel_size, kernel_size=3, padding="same", \
                                           activation=tf.keras.activations.swish)
        self.c2d2 = tf.keras.layers.Conv2D(self.channel_size, kernel_size=3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn2 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, timeemb, training=None):
        self.timeemb = timeemb #[B,c*4]
        input_channel_size = inputs.shape[3]
        if input_channel_size == self.channel_size:
            residual = inputs
        else:
            residual = self.c2d0(inputs)
        x = self.bn1(inputs, training=training)
        x = self.c2d1(x)
        x = x + self.d0(self.timeemb)[:,None,None,:]
        x = tf.keras.activations.swish(self.bn2(x, training=training))
        if training:
            x = tf.nn.dropout(x, rate=0.1)
        x = self.c2d2(x)
        x = tf.keras.layers.Add()([x, residual])
        return x


class Down_block(tf.keras.layers.Layer):
    def __init__(self, channel_size, block_depth, attn_resolutions):
        super(Down_block, self).__init__()
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
        for i in range(self.block_depth):
            x = self.rblock[i](x, self.timeemb, training)
            if x.shape[1] in self.attn_resolutions:
                x = self.attnblock[i](x, training)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x, skips


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

#can't be cloned in existing tf version
class attention_unet(tf.keras.Model):
    def __init__(self, image_size, channel_size, block_depth, embedding_dim=32, \
                 attn_resolutions=[32, 16, 8]):
        super(attention_unet, self).__init__(name='')
        self.channel_size = channel_size
        self.image_size = image_size
        self.block_depth = block_depth
        self.embedding_dim = embedding_dim
        self.attn_resolutions = attn_resolutions

        self.input_dense = tf.keras.layers.Conv2D(self.channel_size[0], kernel_size=1)

        self.timeemb_dense0 = tf.keras.layers.Dense(self.embedding_dim*4, activation=tf.keras.activations.swish)
        self.timeemb_dense1 = tf.keras.layers.Dense(self.embedding_dim*4)

        self.downblocks={}
        for nc in self.channel_size[:-1]:
            self.downblocks[nc] = Down_block(nc, self.block_depth, self.attn_resolutions)
        self.upblocks={}
        for nc in self.channel_size[:-1]:
            self.upblocks[nc] = Up_block(nc, self.block_depth, self.attn_resolutions)

        self.midlocks_rb1=[]
        self.midlocks_rb2=[]
        self.midlocks_attnblock=SelfAttention(self.channel_size[-1])
        for _ in range(self.block_depth):
            self.midlocks_rb1.append(Resnet_block(self.channel_size[-1]))
            self.midlocks_rb2.append(Resnet_block(self.channel_size[-1]))

        self.output_bn=tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.output_con2d=tf.keras.layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")


    def call(self, input, training=None):
        noisy_images, t=input
        timeemb = get_timestep_embedding(t,self.embedding_dim)
        timeemb = self.timeemb_dense0(timeemb) #[B, ch * 4]
        timeemb = self.timeemb_dense1(timeemb) #[B, ch * 4]

        x = self.input_dense(noisy_images)

        skips = []
        for nc in self.channel_size[:-1]:
            x, skips = self.downblocks[nc](x, skips, timeemb, training)

        for mb_i in self.midlocks_rb1:
            x = mb_i(x, timeemb, training)
        x = self.midlocks_attnblock(x, training)
        for mb_i in self.midlocks_rb2:
            x = mb_i(x, timeemb, training)

        for nc in reversed(self.channel_size[:-1]):
            x, skips = self.upblocks[nc](x, skips, timeemb, training)

        x = tf.keras.activations.swish(self.output_bn(x, training=training))
        x = self.output_con2d(x)
        return x
