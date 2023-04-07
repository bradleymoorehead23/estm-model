from functions import *
import gc

seg_len = 222264

df = pd.read_csv('/scratch/gpfs/bcm2/deam_predictions.csv')
df = df.set_index('song_id')

true_df = pd.read_csv('/scratch/gpfs/bcm2/deam_split.csv')
true_df = true_df.loc[:, ['song_id', 'Q1', 'Q2', 'Q3', 'Q4']].set_index('song_id')

train_df = pd.read_csv('/scratch/gpfs/bcm2/estm_train_exp.csv')
test_df = pd.read_csv('/scratch/gpfs/bcm2/estm_test_exp.csv')

max_pct_zero = 0.5
use_argmax = True

X_train = np.zeros((train_df.shape[0], seg_len, 1))
dev_train = np.zeros(train_df.shape[0], dtype=bool)
X_test = np.zeros((test_df.shape[0], seg_len, 1))
dev_test = np.zeros(test_df.shape[0], dtype=bool)

S_train = np.zeros((train_df.shape[0], 2, 4))
S_test = np.zeros((test_df.shape[0], 2, 4))

for j, i in enumerate(train_df.index):
    content, style, dev = train_df.loc[i]
    dev_train[j] = dev
    song_id = int(style.split('-')[0])
    content = f'/scratch/gpfs/bcm2/DEAM_standard_10s/{content}'
    style = f'data\\DEAM_standard_10s\\{style}'
    seg, sr = lb.load(content)
    X_train[j, :, 0] = seg
    style_embed = df.loc[style].to_numpy()
    if use_argmax:
        temp = np.zeros(4)
        temp[np.argmax(style_embed)] = 1
        style_embed = temp
    S_train[j, 0, :] = style_embed # predicted
    S_train[j, 1, :] = true_df.loc[song_id].to_numpy() # true
    
for j, i in enumerate(test_df.index):
    content, style, dev = test_df.loc[i]
    dev_test[j] = dev
    song_id = int(style.split('-')[0])
    content = f'/scratch/gpfs/bcm2/DEAM_standard_10s/{content}'
    style = f'data\\DEAM_standard_10s\\{style}'
    seg, sr = lb.load(content)
    X_test[j, :, 0] = seg
    style_embed = df.loc[style].to_numpy()
    if use_argmax:
        temp = np.zeros(4)
        temp[np.argmax(style_embed)] = 1
        style_embed = temp
    S_test[j, 0, :] = style_embed # predicted
    S_test[j, 1, :] = true_df.loc[song_id].to_numpy() # true

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

train_variance = 0.04383824434697565

model_list = [
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v4.h5', 16, 0.25),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v5.h5', 16, 0.5),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v6.h5', 16, 0.75),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v7.h5', 16, 1.),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v8.h5', 16, 1.25),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v9.h5', 16, 1.5),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v10.h5', 20, 0.25),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v11.h5', 20, 0.5),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v12.h5', 20, 0.75),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v13.h5', 20, 1.),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_21_v14.h5', 20, 1.25),
    ('/scratch/gpfs/bcm2/estm_v1_models/estm_model_2023_03_22_v0.h5', 20, 1.5)
]

mer = keras.models.load_model('/scratch/gpfs/bcm2/mer_models/mer_model_2023_02_12_v7.h5')

metrics = {
    'model': [],
    'train_recon': [],
    'train_pred_emotion': [],
    'train_true_emotion': [],
    'train_content_dev': [],
    'test_recon': [],
    'test_pred_emotion': [],
    'test_true_emotion': [],
    'test_content_dev': []
}
for model in model_list:
    path, latent_dim, beta = model
    trainer = ESTMTrainer(
        train_variance, 
        get_vqvae,
        latent_dim=latent_dim,
        num_embeddings=1024,
        beta=beta
    )
    trainer.build([(None, seg_len, 1), (None, 1, 4)])
    trainer.load_weights(path)
    metrics['model'].append(path)
    
    # train
    X_pred = trainer.predict(
        [X_train, S_train[:, 0, :].reshape((-1, 1, 4))],
        verbose=0
    )
    train_recon = tf.reduce_mean((X_train - X_pred)**2/train_variance).numpy()
    metrics['train_recon'].append(train_recon)
    
    feat = np.zeros((X_pred.shape[0], mels, ts, 1))
    for i in range(feat.shape[0]):
        spec, _, _, _ = audio_to_feat(X_pred[i, :, 0])
        feat[i, :, :, 0] = spec
    mer_pred = mer.predict(feat)
    m = keras.metrics.CategoricalAccuracy()
    m.update_state(S_train[:, 0, :], mer_pred)
    pred_emo = m.result().numpy()
    m = keras.metrics.CategoricalAccuracy()
    m.update_state(S_train[:, 1, :], mer_pred)
    true_emo = m.result().numpy()
    metrics['train_pred_emotion'].append(pred_emo)
    metrics['train_true_emotion'].append(true_emo)
    
    X_dev = X_train[dev_train, :, :]
    dev_train_score = np.zeros(X_dev.shape[0])
    for i in range(X_dev.shape[0]):
        X = X_dev[i, :, :]
        X = np.tile(X, (4, 1, 1))
        S = np.eye(4).reshape((4, 1, 4))
        X_pred = trainer.predict([X, S], verbose=0)
        X_mean = X_pred.mean(axis=0)
        dev_train_score[i] = tf.reduce_mean(
            (X_pred - X_mean)**2
        ).numpy()
    metrics['train_content_dev'].append(dev_train_score.mean())
    
    # test
    X_pred = trainer.predict(
        [X_test, S_test[:, 0, :].reshape((-1, 1, 4))],
        verbose=0
    )
    test_recon = tf.reduce_mean((X_test - X_pred)**2/train_variance).numpy()
    metrics['test_recon'].append(test_recon)
    
    feat = np.zeros((X_pred.shape[0], mels, ts, 1))
    for i in range(feat.shape[0]):
        spec, _, _, _ = audio_to_feat(X_pred[i, :, 0])
        feat[i, :, :, 0] = spec
    mer_pred = mer.predict(feat)
    m = keras.metrics.CategoricalAccuracy()
    m.update_state(S_test[:, 0, :], mer_pred)
    pred_emo = m.result().numpy()
    m = keras.metrics.CategoricalAccuracy()
    m.update_state(S_test[:, 1, :], mer_pred)
    true_emo = m.result().numpy()
    metrics['test_pred_emotion'].append(pred_emo)
    metrics['test_true_emotion'].append(true_emo)
    
    X_dev = X_test[dev_test, :, :]
    dev_test_score = np.zeros(X_dev.shape[0])
    for i in range(X_dev.shape[0]):
        X = X_dev[i, :, :]
        X = np.tile(X, (4, 1, 1))
        S = np.eye(4).reshape((4, 1, 4))
        X_pred = trainer.predict([X, S], verbose=0)
        X_mean = X_pred.mean(axis=0)
        dev_test_score[i] = tf.reduce_mean(
            (X_pred - X_mean)**2
        ).numpy()
    metrics['test_content_dev'].append(dev_test_score.mean())

df = pd.DataFrame(metrics)
df.to_csv('/scratch/gpfs/bcm2/estm_v1_test_results.csv', index=False)