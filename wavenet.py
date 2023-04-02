import tensorflow as tf
import numpy as np

class DilatedConv1d(tf.keras.layers.Layer):
    """Custom implementation of dilated convolution 1D 
    because of the issue https://github.com/tensorflow/tensorflow/issues/26797.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_rate):
        """Initializer.
        Args:
            in_channels: int, input channels.
            out_channels: int, output channels.
            kernel_size: int, size of the kernel.
            dilation_rate: int, dilation rate.
        """
        super(DilatedConv1d, self).__init__()
        self.dilations = dilation_rate

        init = tf.keras.initializers.GlorotUniform()
        self.kernel = tf.Variable(
            init([kernel_size, in_channels, out_channels], dtype=tf.float32),
            trainable=True)
        self.bias = tf.Variable(
            tf.zeros([1, 1, out_channels], dtype=tf.float32),
            trainable=True)

    def call(self, inputs):
        """Pass to dilated convolution 1d.
        Args:
            inputs: tf.Tensor, [B, T, Cin], input tensor.
        Returns:
            outputs: tf.Tensor, [B, T', Cout], output tensor.
        """
        conv = tf.nn.conv1d(
            inputs, self.kernel, 1, padding='SAME', dilations=self.dilations)
        #bias = tf.broadcast_to(self.bias, conv.shape)
        #print('conv shape: ',conv.shape)
        #print('broadcast bias shape: ',bias.shape)
        return conv + self.bias


class Block(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, channels, kernel_size, dilation, last=False):
        """Initializer.
        Args:
            channels: int, basic channel size.
            kernel_size: int, kernel size of the dilated convolution.
            dilation: int, dilation rate.
            last: bool, last block or not.
        """
        super(Block, self).__init__()
        self.channels = channels
        self.last = last

        self.proj_embed = tf.keras.layers.Dense(channels)
        self.conv = DilatedConv1d(
            channels, channels * 2, kernel_size, dilation)
        self.proj_mel = tf.keras.layers.Conv1D(channels * 2, 1)

        if not last:
            self.proj_res = tf.keras.layers.Conv1D(channels, 1)
        self.proj_skip = tf.keras.layers.Conv1D(channels, 1)

    def call(self, inputs, embedding, mel):
        """Pass wavenet block.
        Args:
            inputs: tf.Tensor, [B, T, C(=channels)], input tensor.
            embedding: tf.Tensor, [B, E], embedding tensor for noise schedules.
            mel: tf.Tensor, [B, T // hop, M], mel-spectrogram conditions.
        Returns:
            residual: tf.Tensor, [B, T, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, C], output tensor for skip connection.
        """
        # [B, C]
        embedding = self.proj_embed(embedding)
        # [B, T, C]
        x = inputs + embedding[:, None]
        # [B, T, Cx2]
        #print('mel shape: ',mel.shape)
        #print('proj_mel shape: ',self.proj_mel(mel).shape)
        #print('conv(x) shape: ',self.conv(x).shape)
        x = self.conv(x) + self.proj_mel(mel)
        # [B, T, C]
        context = tf.math.tanh(x[..., :self.channels])
        gate = tf.math.sigmoid(x[..., self.channels:])
        x = context * gate
        # [B, T, C]
        residual = (self.proj_res(x) + inputs) / 2 ** 0.5 if not self.last else None
        skip = self.proj_skip(x)
        return residual, skip



class WaveNet(tf.keras.Model):
    """WaveNet structure.
    """
    def __init__(self):
        """Initializer.
        Args:
            self: Config, model selfuration.
        """
        super(WaveNet, self).__init__()
        self.leak = 0.4

        # embdding self
        self.embedding_size = 128
        self.embedding_proj = 512
        self.embedding_layers = 2
        self.embedding_factor = 4

        # upsampler self
        #self.upsample_stride = [32, 1]
        #self.upsample_kernel = [32, 2]
        self.upsample_stride = [16, 1]
        self.upsample_kernel = [32, 3]
        self.upsample_layers = 2
        # computed hop size
        self.hop = self.upsample_stride[0] ** self.upsample_layers

        # block self
        self.channels = 64
        self.kernel_size = 3
        self.dilation_rate = 2
        self.num_layers = 30
        self.num_cycles = 3

        # noise schedule
        self.iter = 20                  # 20, 40, 50
        self.noise_policy = 'linear'
        self.noise_start = 1e-4
        self.noise_end = 0.05           # 0.02 for 200
        # signal proj
        self.proj = tf.keras.layers.Conv1D(self.channels, 1)
        # embedding
        self.embed = self.embedding(self.iter)
        self.proj_embed = [
            tf.keras.layers.Dense(self.embedding_proj)
            for _ in range(self.embedding_layers)]
        # mel-upsampler
        self.upsample = [
            tf.keras.layers.Conv2DTranspose(
                1,
                self.upsample_kernel,
                self.upsample_stride,
                padding='same')
            for _ in range(self.upsample_layers)]
        # wavenet blocks
        self.blocks = []
        layers_per_cycle = self.num_layers // self.num_cycles
        for i in range(self.num_layers):
            dilation = self.dilation_rate ** (i % layers_per_cycle)
            self.blocks.append(
                Block(
                    self.channels,
                    self.kernel_size,
                    dilation,
                    last=i == self.num_layers - 1))  
        # for output
        self.proj_out = [
            tf.keras.layers.Conv1D(self.channels, 1, activation=tf.nn.relu),
            tf.keras.layers.Conv1D(1, 1)]

    def call(self, signal, timestep, mel):
        """Generate output signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], int, timesteps of current markov chain.
            mel: tf.Tensor, [B, T // hop, M], mel-spectrogram.
        Returns:
            tf.Tensor, [B, T], generated.
        """
        #s=signal.numpy()
        #if np.isnan(s).any():
            #print(f'signal Array contains NaN values.')
        # [B, T, C(=channels)] 8,128,64
        x = tf.nn.relu(self.proj(signal[..., None]))
        # [B, E']
        embed = tf.gather(self.embed, timestep - 1)
        # [B, E] 8,512
        for proj in self.proj_embed:
            embed = tf.nn.swish(proj(embed))
        # [B, T, M, 1], treat as 2D tensor. 8,4(T/hop),80,1
        mel = mel[..., None]
        #print('2D tensor mel shape: ',mel.shape)
        for upsample in self.upsample:
            mel = tf.nn.leaky_relu(upsample(mel), self.leak)
        #print('upsample mel shape: ',mel.shape)
        # [B, T, M] 8,128,80
        mel = tf.squeeze(mel, axis=-1)
        #print('[B, T, M] mel shape: ',mel.shape)

        context = []
        for block in self.blocks:
            # [B, T, C], [B, T, C]
            #print('mel shape: ',mel.shape)
            #print('embed shape: ',embed.shape)
            #print('x shape: ',x.shape)
            x, skip = block(x, embed, mel)
            context.append(skip)
        # [B, T, C]
        scale = self.num_layers ** 0.5
        context = tf.reduce_sum(context, axis=0) / scale
        # [B, T, 1]
        for proj in self.proj_out:
            context = proj(context)
        # [B, T]
        #s=context.numpy()
        #if np.isnan(s).any():
            #print(f'context Array contains NaN values.')
        #s=tf.squeeze(context, axis=-1).numpy()
        #if np.isnan(s).any():
            #print(f'tf.squeeze(context, axis=-1) Array contains NaN values.')    
        return tf.squeeze(context, axis=-1)

    def embedding(self, iter):
        """Generate embedding.
        Args:
            iter: int, maximum iteration.
        Returns:
            tf.Tensor, [iter, E(=embedding_size)], embedding vectors.
        """
        # [E // 2]
        logit = tf.linspace(0., 1., self.embedding_size // 2)
        exp = tf.pow(10, logit * self.embedding_factor)
        # [iter]
        timestep = tf.range(1, iter + 1)
        # [iter, E // 2]
        comp = exp[None] * tf.cast(timestep[:, None], tf.float32)
        # [iter, E]
        return tf.concat([tf.sin(comp), tf.cos(comp)], axis=-1)