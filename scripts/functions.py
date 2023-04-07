from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa as lb
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers
from keras.layers import GRU
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, Conv1DTranspose, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D
from keras.regularizers import L2
from datetime import date

mels   = 128
ts     = 435

def audio_to_feat(audio):
    '''
    audio: 1D np.array of an audio signal.
    
    This function transforms audio into a standardized log-mel-spectrogram.
    '''
    fft = lb.feature.melspectrogram(y=audio, n_fft=1024)
    ref = np.max(fft)
    db_fft = lb.power_to_db(fft, ref=ref)
    
    mean = db_fft.mean()
    std = db_fft.std()
    
    db_norm = (db_fft - mean)/std
    return db_norm, mean, std, ref

def feat_to_audio(feat, mean, std, ref):
    '''
    feat: 2D np.array of a standardized log-mel-spectrogram.
    mean: float; the mean derived from audio_to_feat().
    std: float; the standard deviation derived from audio_to_feat().
    
    This function transforms a 2D spectrogram into an audio signal.
    '''
    db_fft = feat*std + mean
    db_fft = db_fft[:, :, 0]
    
    fft = lb.db_to_power(db_fft, ref=ref)
    
    audio = lb.feature.inverse.mel_to_audio(fft, n_fft=1024, hop_length=512)
    return audio

def get_mer(activation, reg):
    '''
    activation: function; defines the activation of the convolutional and dense layers.
    reg: keras.regularizers Regularizer; defines the kernel regularization applied to the dense layers.
    
    Builds MER model.
    '''
    inp = keras.Input(shape=(mels, ts, 1))
    model = keras.models.Sequential()
    model.add(inp)
    model.add(Conv2D(64, 3, activation=activation, padding='same'))
    model.add(Conv2D(64, 3, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(64, 3, activation=activation, padding='same'))
    model.add(Conv2D(64, 3, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(128, 3, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=3, strides=3))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, 3, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=3, strides=3))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, 3, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 6), strides=(3, 6)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, 3, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation=activation, kernel_regularizer=reg))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation=activation, kernel_regularizer=reg))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    return model

class VectorQuantizer(layers.Layer):
    '''
    Paul, S. (2021) Vector-Quanitized Variational Autoencoders [Source code]. https://keras.io/examples/generative/vq_vae/
    
    num_embeddings: int; number of vectors in the latent space.
    embedding_dim: int; dimensionality of the latent space.
    beta: float (optional); weight of the commitment loss with respect to the codebook loss.
    
    Quantization layer of the VQ-VAE framework.
    '''
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.beta = beta
        
        if 'name' in kwargs.keys():
            name = kwargs['name']
        else:
            name = 'embeddings_vqvae'

        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype='float32'
            ),
            trainable=True,
            name=name
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
class VQVAETrainer(keras.Model):
    '''
    Paul, S. (2021) Vector-Quanitized Variational Autoencoders [Source code]. https://keras.io/examples/generative/vq_vae/
    
    train_variance: float; variance of training spectrograms.
    get_vqvae: callable; must return a keras.Model which defines VQ-VAE.
    latent_dim: int (optional); dimensionality of the latent space.
    num_embeddings: int (optional); number of vectors in the latent space.
    beta: float (optional); weight of the commitment loss with respect to the codebook loss.
    spec: float (optional); weight of spectral STFT reconstruction loss with respect to the standard reconstruction loss.
          By default, the weight is zero, so only standard reconstruction loss is considered.
    
    Wrapper for VQ-VAE training. 
    '''
    def __init__(
        self, 
        train_variance, 
        get_vqvae,
        latent_dim=32, 
        num_embeddings=512, 
        beta=0.25, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings, beta=self.beta)

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self.vqvae(x)
            reconstruction_loss = tf.reduce_mean((x - reconstructions)**2)/self.train_variance
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'vqvae_loss': self.vq_loss_tracker.result()
        }
    
    def call(self, x):
        return self.vqvae(x)
    
class ESTMTrainer(keras.Model):
    '''
    Paul, S. (2021) Vector-Quanitized Variational Autoencoders [Source code]. https://keras.io/examples/generative/vq_vae/
    
    train_variance: float; variance of training spectrograms.
    get_vqvae: callable; must return a keras.Model which defines VQ-VAE.
    latent_dim: int (optional); dimensionality of the latent space.
    num_embeddings: int (optional); number of vectors in the latent space.
    beta: float (optional); weight of the commitment loss with respect to the codebook loss.
    
    Wrapper for VQ-VAE training. 
    '''
    def __init__(
        self, 
        train_variance, 
        get_vqvae,
        latent_dim=32, 
        num_embeddings=512, 
        beta=0.25, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings, beta=self.beta)

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self.vqvae(x)
            content_input, style_input = x[0]
            reconstruction_loss = tf.reduce_mean((content_input - reconstructions)**2)/self.train_variance
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'vqvae_loss': self.vq_loss_tracker.result()
        }
    
    def call(self, x):
        return self.vqvae(x)
    
class ESTM2Trainer(keras.Model):
    '''
    Paul, S. (2021) Vector-Quanitized Variational Autoencoders [Source code]. https://keras.io/examples/generative/vq_vae/
    
    train_variance: float; variance of training spectrograms.
    get_vqvae: callable; must return a keras.Model which defines VQ-VAE.
    latent_dim: int (optional); dimensionality of the latent space.
    num_embeddings: int (optional); number of vectors in the latent space.
    beta: float (optional); weight of the commitment loss with respect to the codebook loss.
    beta_e: float (optional); weight of emotion input loss with respect to total loss.
    inference: bool (optional); indicates that the model is being used for inference (rather than training).
    
    Wrapper for VQ-VAE training. 
    '''
    def __init__(
        self, 
        train_variance, 
        get_vqvae,
        latent_dim=32, 
        num_embeddings=512, 
        beta=0.25, 
        beta_e=0.5,
        inference=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.beta_e = beta_e
        self.inference = inference

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings, self.inference, beta=self.beta)

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')
        self.style_loss_tracker = keras.metrics.Mean(name='style_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.style_loss_tracker
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self.vqvae(x)
            content_input, style_input = x[0]
            reconstruction_loss = tf.reduce_mean((content_input - reconstructions)**2)/self.train_variance
            total_loss = reconstruction_loss + sum(self.vqvae.losses[:-1]) + self.beta_e*self.vqvae.losses[-1]

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses[:-1]))
        self.style_loss_tracker.update_state(self.vqvae.losses[-1])

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'vqvae_loss': self.vq_loss_tracker.result(),
            'style_loss': self.style_loss_tracker.result()
        }
    
    def call(self, x):
        return self.vqvae(x)