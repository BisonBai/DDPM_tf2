import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
from unet_clonable_for_voice import *
from wavenet import WaveNet
import math
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import librosa

num_epochs = 2000  # train for at least 100 epochs for good results 


class DiffusionModel(keras.Model):
    def __init__(self, diffusion_steps,):
        super().__init__()     
        self.diffusion_steps = diffusion_steps
        self.plot_diffusion_steps = int(self.diffusion_steps * 0.8)
        self.normalizer = layers.Normalization()  # standard normalization
        self.sr = 16000
        self.fft = 1024
        self.mel = 80
        self.fmin = 0
        self.fmax = 8000
        self.hop = 256
        self.win = 1024
        melfilter = librosa.filters.mel(sr=self.sr, n_fft=self.fft, n_mels=self.mel, fmin=self.fmin, fmax=self.fmax).T
        self.melfilter = tf.convert_to_tensor(melfilter)
        self.network = WaveNet()

        # hyperparameters: betas, alphas
        self.betas = tf.linspace(start=1e-4, stop=0.05, num=self.diffusion_steps)#0.05 for 20
        self.alphas = 1 - self.betas
        self.alphas_prev = tf.pad(self.alphas[:-1], paddings=[[1, 0]], constant_values=1.0)
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)  # alpha_bar in paper
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], paddings=[[1, 0]], constant_values=1.0)
        self.sqrt_one_over_alphas = tf.sqrt(1. / self.alphas)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)
        self.posterior_var = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="noise_loss")
        self.signal_loss_tracker = keras.metrics.Mean(name="signal_loss")
        #self.MSE_noise_loss_tracker = keras.metrics.Mean(name="MSE_noise_loss")
        self.mel_loss_tracker = keras.metrics.Mean(name="mel_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.signal_loss_tracker,self.mel_loss_tracker]


    def diffusion_schedule(self, diffusion_times):
        noise_rates = tf.reshape(tf.gather(self.sqrt_one_minus_alphas_cumprod, diffusion_times), (-1, 1))
        signal_rates = tf.reshape(tf.gather(self.sqrt_alphas_cumprod, diffusion_times), (-1, 1))
        return noise_rates, signal_rates

    def denoise(self, noisy_signal, diffusion_times, noise_rates, signal_rates, logmel):
        network = self.network
        #print(f'noisy_signal shape {noisy_signal.shape}')
        pred_noises = network(noisy_signal, diffusion_times, logmel)
        pred_signal = (noisy_signal - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_signal
    

    def reverse_diffusion(self, data, initial_noise, diffusion_steps):
        """
        Reverse diffusion (sampling)
        Slightly different implementation from the original one.
        """
        signal, logmel= data
        print(f'signal shape {signal.shape}')
        print(f'logmel shape {logmel.shape}')
        print(f'initial_noise shape {initial_noise.shape}')
        num_signals = tf.shape(initial_noise)[0]
        next_noisy_signal = initial_noise
        if np.isnan(next_noisy_signal.numpy()).any():
                print(f'before loop,Array contains NaN values.')
        for step in range(diffusion_steps, 0, -1):  # t = T,...,1
            print(f'step{step}...')
            noisy_signal = next_noisy_signal
            diffusion_times = tf.fill(dims=(num_signals,), value=step)  # timestep t
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)  # get corresponding weights at t
            pred_noises, pred_signal = self.denoise(
                noisy_signal, diffusion_times, noise_rates, signal_rates, logmel)  # get predicted zt, x0
            beta_t = tf.reshape(tf.gather(self.betas, diffusion_times), (-1, 1))
            sqrt_alpha_t = tf.reshape(tf.gather(self.alphas, diffusion_times), (-1, 1))**0.5
            alpha_cumprod_t = tf.reshape(tf.gather(self.alphas_cumprod, diffusion_times), (-1, 1))
            alpha_cumprod_t_minus_1 = tf.reshape(tf.gather(self.alphas_cumprod_prev, diffusion_times), (-1, 1))
            next_noisy_signal = sqrt_alpha_t*(1-alpha_cumprod_t_minus_1)/(1-alpha_cumprod_t)*noisy_signal + \
                beta_t*(alpha_cumprod_t_minus_1**0.5)/(1-alpha_cumprod_t)*pred_signal
            #next_noisy_images = (noisy_images - beta_t / noise_rates * pred_noises) / ((1 - beta_t) ** 0.5)
            noise = tf.random.normal(shape=next_noisy_signal.shape, dtype=next_noisy_signal.dtype)
            posterior_var = tf.reshape(tf.gather(self.posterior_var, diffusion_times), (-1, 1))
            next_noisy_signal = next_noisy_signal + posterior_var**0.5 * noise
            test_loss = self.loss(signal,pred_signal[0])
            print(f'signal_loss {test_loss}')
            pred_logmel = self.mel_fn(pred_signal)
            mel_loss = self.loss(logmel, pred_logmel)
            print("Mel loss:", tf.reduce_mean(mel_loss).numpy())
            
        return pred_signal[0],signal # predicted x0 from noisy image x1



    def generate(self,data,diffusion_steps, initial_noise=None):
        # initla noise (pure noise or conditional) -> images -> denormalized images
        signal, logmel= data
        if not np.any(initial_noise):
            print('no initial_noise')
            initial_noise = tf.random.normal(shape=tf.shape(signal))  # pure noise
        generated_signal = self.reverse_diffusion(data=data,initial_noise=initial_noise, diffusion_steps=diffusion_steps)
        return generated_signal

    def train_step(self, data):
        signal, logmel= data
        self.test_data = data
        noises = tf.random.normal(shape=(tf.shape(signal)))
        diffusion_times = tf.random.uniform(shape=(tf.shape(signal)[0],), minval=0, maxval=self.diffusion_steps, dtype=tf.int32)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noised_signal = signal_rates * signal + noise_rates * noises
        with tf.GradientTape(persistent=True) as tape:
            pred_noises, pred_signal = self.denoise(
                noised_signal, diffusion_times, noise_rates, signal_rates,logmel) 
            pred_logmel = self.mel_fn(pred_signal)
            mel_loss = self.loss(logmel, pred_logmel)
            mel_loss = tf.reduce_mean(mel_loss)
            signal_loss = self.loss(signal, pred_signal)
            noise_loss = tf.reduce_mean(tf.abs(pred_noises - noises))
            #MSE_noise_loss = self.loss(noises, pred_noises)
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.signal_loss_tracker.update_state(signal_loss)
        self.mel_loss_tracker.update_state(mel_loss)
        #self.MSE_noise_loss_tracker.update_state(MSE_noise_loss)
            
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        signal, logmel= data
        """similar to train step, but doesn't update weights but output extra KID to compare real and generated image"""
        noises = tf.random.normal(shape=(tf.shape(signal)))

        # sample uniform random diffusion times, (B,)
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(signal)[0],), minval=0, maxval=self.diffusion_steps, dtype=tf.int32)

        # forward propagation: get the weights/rates and compute weighted average sum of images and noises
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_signal = signal_rates * signal + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_signal = self.denoise(
            noisy_signal, diffusion_times, noise_rates, signal_rates,logmel)
        pred_logmel = self.mel_fn(pred_signal)
        mel_loss = self.loss(logmel, pred_logmel)
        mel_loss = tf.reduce_mean(mel_loss)
        noise_loss = self.loss(noises, pred_noises)
        signal_loss = self.loss(signal, pred_signal)

        self.signal_loss_tracker.update_state(signal_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        self.mel_loss_tracker.update_state(mel_loss)
        return {m.name: m.result() for m in self.metrics}
    
    def mel_fn(self, signal):

        padlen = 512
        # [B, T + win - 1]
        center_pad = tf.pad(signal, [[0, 0], [padlen, padlen - 1]], mode='reflect')
        # [B, T // hop, fft // 2 + 1]

        stft = tf.signal.stft(
            center_pad,
            frame_length=self.win,
            frame_step=self.hop,
            fft_length=self.fft,
            window_fn= tf.signal.hann_window)
        # [B, T // hop, mel]

        mel = tf.abs(stft) @ self.melfilter
        # [B, T // hop, mel]
        logmel = tf.math.log(tf.maximum(mel, 1e-5))
        return logmel


