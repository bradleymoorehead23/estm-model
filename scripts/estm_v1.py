#!/usr/bin/env python
# coding: utf-8

from functions import *

gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0, 'No GPUs available.'

n_segs  = 10095
seg_len = 222264
batch_size = 128

df = pd.read_csv('/scratch/gpfs/bcm2/deam_split.csv')
songs = df['song_id']
test = df['test']
val = df['validation']

df = pd.read_csv('/scratch/gpfs/bcm2/deam_predictions.csv')
df = df.set_index('song_id')

use_argmax = True

max_pct_zero = 0.5
paths = [f'/scratch/gpfs/bcm2/DEAM_standard_10s/{song}-*.wav' for song in songs]
X = np.zeros((n_segs, seg_len, 1))
S = np.zeros((n_segs, 1, 4))
j = 0
for i in range(len(paths)):
    files = glob(paths[i])
    for file in files:
        seg, sr = lb.load(file)
        n = seg.shape[0]
        if ((seg == 0).sum()/n <= max_pct_zero) and (test[i] == 0):
            X[j, :, 0] = seg
            index = file.replace('/', '\\')
            index = index.split('\\')[-1]
            index = f'data\\DEAM_standard_10s\\{index}'
            style_embed = df.loc[index].to_numpy()
            if use_argmax:
                temp = np.zeros(4)
                temp[np.argmax(style_embed)] = 1
                style_embed = temp
            S[j, 0, :] = style_embed
            j += 1
X = X[:j, :, :]
S = S[:j, :, :]

lrelu = keras.layers.LeakyReLU(alpha=0.01)

def get_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(seg_len, 1))
    level_rate = [3, 5, 7]
    levels = []
    for rate in level_rate:
        level = encoder_inputs
        for i in range(rate):
            level = Conv1D(
                latent_dim, 
                3, 
                strides=2, 
                activation=lrelu
            )(level)
        levels.append(level)
    return keras.Model(encoder_inputs, levels, name='encoder')

def get_decoder(latent_dim):
    intermediate_steps = 27782
    intermediate_ch = int(3*latent_dim/2)
    latent_inputs = keras.Input(shape=(intermediate_steps, intermediate_ch))
    ch = int(intermediate_ch/2)
    x = Conv1DTranspose(ch, 3, strides=2, activation=lrelu)(latent_inputs)
    ch = int(ch/2)
    x = Conv1DTranspose(ch, 3, strides=2, activation=lrelu)(x)
    decoder_outputs = Conv1DTranspose(1, 3, strides=2, activation=lrelu, output_padding=1)(x)
    return keras.Model(latent_inputs, decoder_outputs, name='decoder')

def get_vqvae(latent_dim, num_embeddings, beta=0.25):
    vq_layers = [
        VectorQuantizer(num_embeddings, latent_dim, beta=beta, name=f'embeddings_vqvae_{i}') for i in range(3)
    ]
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim + 4)
    content_input = keras.Input(shape=(seg_len, 1))
    style_input = keras.Input(shape=(1, 4))
    
    encoder_outputs = encoder(content_input)
    
    quantized_latents = [
        vq_layer(encoder_output) for vq_layer, encoder_output in zip(vq_layers, encoder_outputs)
    ]
    
    intermediate = int((latent_dim + 4)/2)
    
    bottom = quantized_latents[0]
    bottom_style = tf.tile(style_input, (1, bottom.shape[1], 1))
    bottom = layers.Concatenate(axis=-1)([bottom, bottom_style])
    bottom = Conv1DTranspose(
        intermediate, 3, activation=lrelu, padding='same'
    )(bottom)
    
    middle = quantized_latents[1]
    middle_style = tf.tile(style_input, (1, middle.shape[1], 1))
    middle = layers.Concatenate(axis=-1)([middle, middle_style])
    middle = Conv1DTranspose(
        latent_dim, 3, strides=2, activation=lrelu, output_padding=1
    )(middle)
    middle = Conv1DTranspose(
        intermediate, 3, strides=2, activation=lrelu, output_padding=1
    )(middle)
    
    top    = quantized_latents[2]
    top_style = tf.tile(style_input, (1, top.shape[1], 1))
    top    = layers.Concatenate(axis=-1)([top, top_style])
    top    = Conv1DTranspose(
        latent_dim, 3, strides=2, activation=lrelu
    )(top)
    top    = Conv1DTranspose(
        latent_dim, 3, strides=2, activation=lrelu, output_padding=1
    )(top)
    top    = Conv1DTranspose(
        latent_dim, 3, strides=2, activation=lrelu, output_padding=1
    )(top)
    top    = Conv1DTranspose(
        intermediate, 3, strides=2, activation=lrelu, output_padding=1
    )(top)
    
    intermediate_latents = layers.Concatenate()([bottom, middle, top])
    reconstructions = decoder(intermediate_latents)
    return keras.Model([content_input, style_input], reconstructions, name='vq_vae')

train_variance = np.var(X)
print(f'train_variance={train_variance}')

optimizer = keras.optimizers.Adam(learning_rate=0.001)
callback = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    start_from_epoch=25,
    verbose=1
)

latent_dim = 24 # latent dimension AFTER concatenating emotion input
trainer = ESTMTrainer(
    train_variance, 
    get_vqvae, 
    latent_dim=latent_dim - 4,
    num_embeddings=1024,
    beta=1.5
)
trainer.compile(optimizer=optimizer)
history = trainer.fit(
    [X, S],
    epochs=50,
    batch_size=batch_size,
    callbacks=[callback]
)

today = str(date.today()).replace('-', '_')
i = 0
current = glob(f'/scratch/gpfs/bcm2/estm_v1_models/estm_model_{today}*')
current_versions = []
for item in current:
    string = item[:-3]
    j = -1
    while string[j - 1] != 'v':
        j += -1
    current_versions.append(int(string[j:]))
while i in current_versions:
    i += 1
model_dest = f'/scratch/gpfs/bcm2/estm_v1_models/estm_model_{today}_v{i}.h5'
hist_dest  = f'/scratch/gpfs/bcm2/estm_v1_models/estm_history_{today}_v{i}.csv'
trainer.save_weights(model_dest)
history_df = pd.DataFrame(history.history)
history_df.to_csv(hist_dest)