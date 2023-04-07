from functions import *
import gc

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
    
def get_vqvae(latent_dim, num_embeddings, inference, beta=0.25, **kwargs):
    content_input = keras.Input(shape=(seg_len, 1))
    style_input = keras.Input(shape=(1, 4))
    
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    vq_layers = [
        VectorQuantizer(num_embeddings, latent_dim, beta=beta, name=f'embeddings_vqvae_{i}') for i in range(3)
    ]
    
    encoder_outputs = encoder(content_input)

    bottom, middle, top = [
        vq_layer(encoder_output) for vq_layer, encoder_output in zip(vq_layers, encoder_outputs)
    ]

    intermediate = int(latent_dim/2)

    if inference:
        style_input = keras.Input(shape=(seg_len, 1))
        encoder_outputs = encoder(style_input)
        style_latents = [
            vq_layer(encoder_output) for vq_layer, encoder_output in zip(vq_layers, encoder_outputs)
        ]
        bottom_style, middle_style, top_style = [
            quantized_latents[:, :, -4:] for quantized_latents in style_latents
        ]
    else:
        bottom_style, middle_style, top_style = [
            quantized_latents[:, :, -4:] for quantized_latents in [bottom, middle, top]
        ]
    bottom_style = layers.Softmax(axis=-1)(bottom_style)
    middle_style = layers.Softmax(axis=-1)(middle_style)
    top_style = layers.Softmax(axis=-1)(top_style)
    bottom = layers.Concatenate()([bottom[:, :, :-4], bottom_style])
    middle = layers.Concatenate()([middle[:, :, :-4], middle_style])
    top = layers.Concatenate()([top[:, :, :-4], top_style])
            
    bottom = Conv1DTranspose(
        intermediate, 3, activation=lrelu, padding='same'
    )(bottom)

    middle = Conv1DTranspose(
        latent_dim - 4, 3, strides=2, activation=lrelu, output_padding=1
    )(middle)
    middle = Conv1DTranspose(
        intermediate, 3, strides=2, activation=lrelu, output_padding=1
    )(middle)

    top = Conv1DTranspose(
        latent_dim - 4, 3, strides=2, activation=lrelu
    )(top)
    top = Conv1DTranspose(
        latent_dim - 4, 3, strides=2, activation=lrelu, output_padding=1
    )(top)
    top = Conv1DTranspose(
        latent_dim - 4, 3, strides=2, activation=lrelu, output_padding=1
    )(top)
    top = Conv1DTranspose(
        intermediate, 3, strides=2, activation=lrelu, output_padding=1
    )(top)

    intermediate_latents = layers.Concatenate()([bottom, middle, top])
    reconstructions = decoder(intermediate_latents)
        
    vq_vae = keras.Model([content_input, style_input], reconstructions, name='vq_vae')
    if inference:
        vq_vae.add_loss(lambda: 0)
    else:
        names = [f'{level}_cat_crossentropy' for level in ['bottom', 'middle', 'top']]
        style_losses = [
            keras.losses.CategoricalCrossentropy(name=name)(
                tf.tile(style_input, (1, level.shape[1], 1)), 
                level
            ) for level, name in zip([bottom_style, middle_style, top_style], names)
        ]
        vq_vae.add_loss(sum(style_losses)/3)
    return vq_vae
    
train_variance = np.var(X)
print(f'train_variance={train_variance}')

callback = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    start_from_epoch=25,
    verbose=0,
    restore_best_weights=True
)
stopifnan = keras.callbacks.TerminateOnNaN()

params = [
    (0.75, 20, 0.25),
    (0.75, 20, 0.5),
    (0.75, 20, 1.25),
    (0.75, 20, 1.5)
]

for param_set in params:
    beta_e, latent_dim, beta = param_set
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    trainer = ESTM2Trainer(
        train_variance, 
        get_vqvae, 
        latent_dim=latent_dim,
        num_embeddings=1024,
        beta=beta,
        beta_e=beta_e,
        inference=False
    )

    trainer.compile(optimizer=optimizer)
    history = trainer.fit(
        [X, S],
        epochs=50,
        batch_size=batch_size,
        callbacks=[callback, stopifnan],
        verbose=0
    )
            
    trainer.save_weights(f'/scratch/gpfs/bcm2/estm_v2_grid/estm2_{beta_e}_{latent_dim}_{beta}.h5')
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'/scratch/gpfs/bcm2/estm_v2_grid/estm2_history_{beta_e}_{latent_dim}_{beta}.csv')
            
    del trainer, history, history_df
    gc.collect()

# today = str(date.today()).replace('-', '_')
# i = 0
# current = glob(f'/scratch/gpfs/bcm2/estm_v2_models/estm_model_{today}*')
# current_versions = []
# for item in current:
#     string = item[:-3]
#     j = -1
#     while string[j - 1] != 'v':
#         j += -1
#     current_versions.append(int(string[j:]))
# while i in current_versions:
#     i += 1
# model_dest = f'/scratch/gpfs/bcm2/estm_v2_models/estm_model_{today}_v{i}.h5'
# hist_dest  = f'/scratch/gpfs/bcm2/estm_v2_models/estm_history_{today}_v{i}.csv'
# trainer.save_weights(model_dest)
# history_df = pd.DataFrame(history.history)
# history_df.to_csv(hist_dest)