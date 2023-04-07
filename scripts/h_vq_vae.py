from functions import *
import os
import gc

gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0, 'No GPUs available.'

n_segs = 10095
seg_len = 222264
batch_size = 128

df = pd.read_csv('/scratch/gpfs/bcm2/deam_split.csv')
songs = df['song_id']
test = df['test']
val = df['validation']

max_pct_zero = 0.5
paths = [f'/scratch/gpfs/bcm2/DEAM_standard_10s/{song}-*.wav' for song in songs]
X = np.zeros((n_segs, seg_len, 1))
j = 0
for i in range(len(paths)):
    files = glob(paths[i])
    for file in files:
        seg, sr = lb.load(file)
        n = seg.shape[0]
        if ((seg == 0).sum()/n <= max_pct_zero) and (test[i] == 0):
            X[j, :, 0] = seg
            j += 1
X = X[:j, :, :]

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
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(seg_len, 1))
    
    encoder_outputs = encoder(inputs)
    
    quantized_latents = [
        vq_layer(encoder_output) for vq_layer, encoder_output in zip(vq_layers, encoder_outputs)
    ]
    
    intermediate = int(latent_dim/2)
    
    bottom = Conv1DTranspose(
        intermediate, 3, activation=lrelu, padding='same'
    )(quantized_latents[0])
    
    middle = Conv1DTranspose(
        latent_dim, 3, strides=2, activation=lrelu, output_padding=1
    )(quantized_latents[1])
    middle = Conv1DTranspose(
        intermediate, 3, strides=2, activation=lrelu, output_padding=1
    )(middle)
    
    top    = Conv1DTranspose(
        latent_dim, 3, strides=2, activation=lrelu
    )(quantized_latents[2])
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
    return keras.Model(inputs, reconstructions, name='vq_vae')

train_variance = np.var(X)
print(f'train_variance={train_variance}')

callback = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    start_from_epoch=25,
    verbose=1
)
stopifnan = keras.callbacks.TerminateOnNaN()
    
for latent_dim in [16, 20, 24, 28]:
    for num_embeddings in [1024]:
        for beta in [0.25, 0.75, 1.5]:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            trainer = VQVAETrainer(
                train_variance, 
                get_vqvae, 
                latent_dim=latent_dim,
                num_embeddings=num_embeddings,
                beta=beta
            )
            trainer.compile(optimizer=optimizer)
            history = trainer.fit(
                X,
                epochs=50,
                batch_size=batch_size,
                callbacks=[callback, stopifnan],
                verbose=0
            )
            trainer.save_weights(f'/scratch/gpfs/bcm2/hvqvae_grid/hvqvae_{latent_dim}_{num_embeddings}_{beta}.h5')
            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv(f'/scratch/gpfs/bcm2/hvqvae_grid/hvqvae_history_{latent_dim}_{num_embeddings}_{beta}.csv')
            del trainer, history, hist_df
            gc.collect()

# today = str(date.today()).replace('-', '_')
# i = 0
# current = glob(f'/scratch/gpfs/bcm2/vqvae_models/hvqvae_model_{today}*')
# current_versions = []
# for item in current:
#     string = item[:-3]
#     j = -1
#     while string[j - 1] != 'v':
#         j += -1
#     current_versions.append(int(string[j:]))
# while i in current_versions:
#     i += 1
# model_dest = f'/scratch/gpfs/bcm2/vqvae_models/hvqvae_model_{today}_v{i}.h5'
# hist_dest  = f'/scratch/gpfs/bcm2/vqvae_models/hier_history_{today}_v{i}.csv'
# trainer.save_weights(model_dest)
# history_df = pd.DataFrame(history.history)
# history_df.to_csv(hist_dest)