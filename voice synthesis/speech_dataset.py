import os
import librosa
import numpy as np
import tensorflow as tf

class Speech_dataset:
    def __init__(self,data_dir='/home/lqm/Diffusion-Bison/data_long_16'):
        self.sr = 16000
        self.fft = 1024
        self.mel = 80
        self.fmin = 0
        self.fmax = 8000
        self.hop = 256
        self.win = 1024
        self.data_dir = data_dir
        self.rawset, self.info = self.load_data()
        data_size = self.rawset.cardinality().numpy()
        for signal in self.rawset.take(1):
            signal_shape = signal.shape
        print("dataset size:", data_size)
        print("sample size:", signal_shape)
        # [fft // 2 + 1, mel]
        melfilter = librosa.filters.mel(
            sr=self.sr, n_fft=self.fft, n_mels=self.mel, fmin=self.fmin, fmax=self.fmax).T
        self.melfilter = tf.convert_to_tensor(melfilter)
        self.normalized = None
    
    @staticmethod    
    def _load_audio(path):
        raw = tf.io.read_file(path)
        
        audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
        return tf.squeeze(audio, axis=-1)

    def load_data(self):
        # generate file lists

        files = tf.data.Dataset.from_tensor_slices(
            [os.path.join(self.data_dir, n) for n in os.listdir(self.data_dir)])
        # read audio
        return files.map(Speech_dataset._load_audio), None
        
    def normalizer(self, frames=16000):
        def normalize(speech):
            nonlocal frames
            #tf.print('ori speech sahpe', speech.shape)
            frames = frames // self.hop * self.hop
            start = tf.random.uniform(
                (), 0, tf.shape(speech)[0] - frames, dtype=tf.int32)
            #tf.print('convert speech sahpe', speech[start:start + frames].shape)
            return speech[start:start + frames]
        return normalize
        
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
        return signal, logmel
        
    def audio_dataset(self):
    
        self.normalized = self.rawset \
            .map(self.normalizer(6400)) \
            .batch(8) \
            .map(self.mel_fn)
        #print('all good')
        return self.normalized


