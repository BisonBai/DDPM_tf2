'''
This file has some bugs. It's better to use diffusion_model.py
'''
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
from unet_clonable import *
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from tensorflow.keras.backend import eval
 

# data
dataset_name = "oxford_flowers102"
#dataset_name = 'tf_flowers'
#dataset_name = 'cifar10'
#dataset_name = 'cars196'

dataset_repetitions = 5
num_epochs = 100  # train for at least 100 epochs for good results 
image_size = 64   ## for cifar10 , change the image size to 32 
DEFAULT_DTYPE = tf.float32
image_save_file = './result'
batch_size = 64 ## change to 128 for cifar10

# KID = Kernel Inception Distance
kid_image_size = 75
kid_diffusion_steps = 5

# optimization
ema = 0.99
learning_rate = 1e-3
weight_decay = 1e-4



def preprocess_image(data):
    # center crop image
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(split, with_info=False):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    if with_info:
        ds, info = tfds.load(dataset_name, split=split, shuffle_files=True, with_info=with_info)
        loader = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().\
            repeat(dataset_repetitions).shuffle(10 * batch_size).\
            batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds, loader, info
    else:
        return (
            tfds.load(dataset_name, split=split, shuffle_files=True)
            .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .repeat(dataset_repetitions)
            .shuffle(10 * batch_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

class FID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # FID is estimated per batch and is averaged across batches
        self.fid_tracker = keras.metrics.Mean(name="fid_tracker")
        self.model_i = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    
    def result(self):
        return self.fid_tracker.result()

    def reset_state(self):
        self.fid_tracker.reset_state()
        
     # calculate frechet inception distance
    def update_state(self, images1, images2):
 
        # resize images
        images1 = tf.image.resize(images1, (299,299) , method = 'nearest')
        images2 = tf.image.resize(images2, (299,299), method = 'nearest')
        # pre-process images
        images1 = preprocess_input(images1)
        images2 = preprocess_input(images2)
        
    # calculate activations
        act1 = self.model_i(images1)
        act2 = self.model_i(images2)
        
        #mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        #mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means        
        #ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        #covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        #if iscomplexobj(covmean):
            #covmean = covmean.real
        # calculate score
        #fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        
        ## above code needs to be changed for tensorflow as below
        
        
        # calculate mean and covariance statistics
        mu1 = tf.math.reduce_mean(act1 , axis =0 ) ## 2048,
        mu2 = tf.math.reduce_mean(act2, axis =0 )  ## 2048,
        
        
        
        ##
        sigma1 = tfp.stats.covariance(act1)  ## 2048 x 2048 
        sigma2 = tfp.stats.covariance(act2)  ## 2048 x 2048 
        
        
        
        ssdiff = tf.math.reduce_sum(tf.math.square(tf.math.subtract(mu1, mu2))) ## this is ok . single value 
        

        covmean =  tf.math.sqrt(tf.tensordot(sigma1, sigma2 , 1)) ## has NaN values 
        
        covmean = tf.math.real(covmean) ## this has some nan 2048 x 2048
        

        sigma_sum = tf.math.add(sigma1 , sigma2 )   ## 2048 x 2048 
        
        
        prod = tf.math.add(sigma_sum , tf.math.multiply(-2.0, covmean)) ## 2048 x 2048 (contains some nan)
        
        
        fid = tf.math.add(ssdiff , tf.linalg.trace(prod)) ## getting nan ..linalg giving nan
        
        
        # update the average FID estimate
        self.fid_tracker.update_state(fid)
        #return fid   

class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
        

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
                batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth, diffusion_steps):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.plot_diffusion_steps = int(self.diffusion_steps * 0.8)
        self.normalizer = layers.Normalization()  # standard normalization
        self.network = get_network(image_size, widths, block_depth)  # unet model
        self.ema_network = get_network(image_size, widths, block_depth)  # copy of model, for evaluation only
        self.ema_network.set_weights(self.network.get_weights())
        self.image_size = image_size
        

        # hyperparameters: betas, alphas
        self.betas = tf.linspace(start=1e-4, stop=0.02, num=self.diffusion_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)  # alpha_bar in paper
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], paddings=[[1, 0]], constant_values=1.0)
        self.sqrt_one_over_alphas = tf.sqrt(1. / self.alphas)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)
        self.posterior_var = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")  # Kernel inception distance, image quality metric, simpler to implement than FID
        #self.fid = FID(name = "fid") ##  
        
    @property
    def metrics(self):
        #return [self.noise_loss_tracker, self.image_loss_tracker, self.kid , self.fid] 
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        """
        This function will tell us the noise levels and signal levels of the noisy image
        corresponding to the actual diffusion time. One can think of these noise and signal levels as weights.
        f: diffusion times -> noise_rates(sqrt_one_minus_alpha_cumprod), signal_rates(sqrt_alpha_cumprod)
        params:
            diffusion_times, (B,)
                these are uniformly sampled time for each image in a batch, e.g. [3,100,65,200] if B=4
        returns:
            noise_rates (B,1,1,1)
            signal_rates (B,1,1,1)
        """
        noise_rates = tf.reshape(tf.gather(self.sqrt_one_minus_alphas_cumprod, diffusion_times), (-1, 1, 1, 1))
        signal_rates = tf.reshape(tf.gather(self.sqrt_alphas_cumprod, diffusion_times), (-1, 1, 1, 1))
        return noise_rates, signal_rates

    def denoise(self, noisy_images, diffusion_times, noise_rates, signal_rates, training):
        """
        This function will call the main network to predict noise at the given time for the given noisy image
        and predict the denoised image as well.
        f: xt -> zt, x0
        params:
            noisy_images: xt, shape=(B, h, w, c)
            diffusion_times: t, shape=(B,)
            noise_rates: sqrt_one_minus_alpha_cumprod, shape=(B,1,1,1)
            signal_rates: sqrt_alpha_cumprod, shape=(B,1,1,1)
            training: bool
        returns:
            pred_noises: zt, shape=(B, h, w, c)
            pred_images: x0, shape=(B, h, w, c)
        """
        if training:
            network = self.network  # training model, update gradients via GD
        else:
            network = self.ema_network  # exponential moving average of weights (used for evaluation)

        # predict noise component and calculate the image using it
        pred_noises = network([noisy_images, diffusion_times], training=training)  # predicted zt
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates  # predicted x0

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps , gen_plot):
        """
        Reverse diffusion (sampling)
        Slightly different implementation from the original one.
        """
        num_images = initial_noise.shape[0]  # batch size

        next_noisy_images = initial_noise
        
        sr =[]
        nr = []
        d_step =[]
        predicted_noise = []
        
        for step in range(diffusion_steps, 0, -1):  # t = T,...,1
            
            noisy_images = next_noisy_images

            # separate the current noisy image to its components: noise zt and denoised image x0
            diffusion_times = tf.fill(dims=(num_images,), value=step)  # timestep t
            
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)  # get corresponding weights at t
            
            pred_noises, pred_images = self.denoise(
            noisy_images, diffusion_times, noise_rates, signal_rates, training=False) 
            
            ####
            nr.append(tf.math.reduce_mean(noise_rates).numpy())
            sr.append(tf.math.reduce_mean(signal_rates).numpy())
            d_step.append(step) 
            predicted_noise.append(tf.math.reduce_mean(pred_noises).numpy())
            ####
            
            # get predicted zt, x0
            beta_t = tf.reshape(tf.gather(self.betas, diffusion_times), (-1, 1, 1, 1))
            sqrt_alpha_t = tf.reshape(tf.gather(self.alphas, diffusion_times), (-1, 1, 1, 1))**0.5
            alpha_cumprod_t = tf.reshape(tf.gather(self.alphas_cumprod, diffusion_times), (-1, 1, 1, 1))
            alpha_cumprod_t_minus_1 = tf.reshape(tf.gather(self.alphas_cumprod_prev, diffusion_times), (-1, 1, 1, 1))
            next_noisy_images = sqrt_alpha_t*(1-alpha_cumprod_t_minus_1)/(1-alpha_cumprod_t)*noisy_images + \
                beta_t*(alpha_cumprod_t_minus_1**0.5)/(1-alpha_cumprod_t)*pred_images
            #next_noisy_images = (noisy_images - beta_t / noise_rates * pred_noises) / ((1 - beta_t) ** 0.5)
            noise = tf.random.normal(shape=next_noisy_images.shape, dtype=next_noisy_images.dtype)
            posterior_var = tf.reshape(tf.gather(self.posterior_var, diffusion_times), (-1, 1, 1, 1))
            next_noisy_images = next_noisy_images + posterior_var**0.5 * noise
            ##
            
        ###    
        
        if gen_plot:
            
            plt.figure(figsize =(10,5))
            plt.plot(d_step ,sr,  label ="signal rate")
            plt.title(" Mean Signal Rate Vs Diffusion step")
            plt.figure(figsize =(10,5))
            plt.plot(d_step ,nr,  label ="noise rate")
            plt.title(" Mean Noise Rate Vs Diffusion step")
            plt.figure(figsize =(10,5))
            plt.plot(d_step ,predicted_noise,  label ="predicted noise ")
            plt.title(" Mean Predicted noise Vs Diffusion step")
        ###
                   
        return pred_images  # predicted x0 from noisy image x1



    def generate(self, num_images, diffusion_steps,  gen_plot , initial_noise=None ):
        # initla noise (pure noise or conditional) -> images -> denormalized images
        if not initial_noise:
            initial_noise = tf.random.normal(shape=(num_images, self.image_size, self.image_size, 3))  # pure noise
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps , gen_plot )
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)  # （B, h, w, c)
        noises = tf.random.normal(shape=(images.shape[0], self.image_size, self.image_size, 3))  # (B, h, w, c)

        # sample uniform random diffusion times, (B,)
        diffusion_times = tf.random.uniform(
            shape=(images.shape[0],), minval=0, maxval=self.diffusion_steps, dtype=tf.int32)

        # forward propagation: get the weights/rates and compute weighted average sum of images and noises
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # print(signal_rates.shape, images.shape, noise_rates.shape)
        noisy_images = signal_rates * images + noise_rates * noises

        # compute loss, gradients and update model weights
        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, diffusion_times, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric (evaluation)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights, ema=0.999, like GD with momemtum
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]} ## 3/4 V

    def test_step(self, images):
        """similar to train step, but doesn't update weights but output extra KID to compare real and generated image"""
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(images.shape[0], self.image_size, self.image_size, 3))

        # sample uniform random diffusion times, (B,)
        diffusion_times = tf.random.uniform(
            shape=(images.shape[0],), minval=0, maxval=self.diffusion_steps, dtype=tf.int32)

        # forward propagation: get the weights/rates and compute weighted average sum of images and noises
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, diffusion_times, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # computationally demanding, kid_diffusion_steps has to be small
        
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps, gen_plot =False , initial_noise=None  
        )
        self.kid.update_state(images, generated_images)
        #self.fid.update_state(images1 = images, images2= generated_images)
        
        
        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, plot_diffusion_steps = None, \
                    image_save_file=image_save_file, num_rows=3, num_cols=6 , gen_plot = False):
        if epoch % 10 == 0:
            # plot random generated images for visual evaluation of generation quality
            if plot_diffusion_steps == None:
                plot_diffusion_steps = self.plot_diffusion_steps
            generated_images = self.generate(
                num_images=num_rows * num_cols,
                diffusion_steps=plot_diffusion_steps , gen_plot = gen_plot
            )
            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(generated_images[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.savefig("%s/diffusion_steps_%d_epoch_%d.jpg" % (image_save_file, self.diffusion_steps, epoch))
            plt.show()
            plt.close()
            
    
    

   
    
    


