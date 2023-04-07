#!/usr/bin/env python
# coding: utf-8

from functions import *

assert len(tf.config.list_physical_devices('GPU')) > 0, 'No GPUs available.'

file_df = pd.read_csv('/scratch/gpfs/bcm2/mer_taffc_10s_split.csv')

n_segs = 2696
dur  = '10s'

max_pct_zero = 0.5
paths = file_df['path'].tolist()
qs    = file_df['quadrant']
test  = file_df['test']
val   = file_df['validation']
for i, path in enumerate(paths):
    split = path.split('\\')
    direc = '/scratch/gpfs/bcm2'
    folder = f'MER_taffc_standard_{dur}'
    split[0] = direc
    split[1] = folder
    split[-1] = split[-1][:-4] + '-*'
    paths[i] = '/'.join(split)
X = np.zeros((n_segs, mels, ts, 1))
y = np.zeros((n_segs, 4))
test_val_train = np.zeros((n_segs, 3), dtype=bool)
j = 0 # file counter
for i in range(len(paths)):
    files = glob(paths[i])
    for file in files:
        seg, sr = lb.load(file)
        n = seg.shape[0]
        if (seg == 0).sum()/n <= max_pct_zero:
            db_norm, _, _ = audio_to_feat(seg)
            X[j, :, :, 0] = db_norm
            q = qs[i]
            if q == 'Q1':
                target = [1, 0, 0, 0]
            elif q == 'Q2':
                target = [0, 1, 0, 0]
            elif q == 'Q3':
                target = [0, 0, 1, 0]
            else:
                target = [0, 0, 0, 1]
            y[j, :] = target
            test_val_train[j, 0] = test[i]
            test_val_train[j, 1] = val[i]
            test_val_train[j, 2] = 1 - max(test[i], val[i])
            j += 1
X = X[:j, :, :, :]
y = y[:j, :]
test_val_train = test_val_train[:j, :]

lrelu = keras.layers.LeakyReLU(alpha=0.01)
l2 = L2(l2=0.001)
batch_size = 128

model = get_mer(lrelu, l2)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

X_train = X[test_val_train[:, 2], :, :, :]
X_val   = X[test_val_train[:, 1], :, :, :]
X_test  = X[test_val_train[:, 0], :, :, :]

y_train = y[test_val_train[:, 2], :]
y_val   = y[test_val_train[:, 1], :]
y_test  = y[test_val_train[:, 0], :]

# for final model (X_train+X_val)
X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])

# for training with validation
# callback = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=1, 
#     restore_best_weights=True,
#     start_from_epoch=25
# )
# history = model.fit(
#     x=X_train,
#     y=y_train,
#     epochs=50,
#     batch_size=batch_size,
#     callbacks=[callback],
#     validation_data=(X_val, y_val)
# )

# for final model (X_train+X_val)
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=27,
    batch_size=batch_size
)

today = str(date.today()).replace('-', '_')
i = 0
current = glob(f'/scratch/gpfs/bcm2/mer_models/mer_model_{today}*')
current_versions = []
for item in current:
    string = item[:-3]
    j = -1
    while string[j - 1] != 'v':
        j += -1
    current_versions.append(int(string[j:]))
while i in current_versions:
    i += 1
model_dest = f'/scratch/gpfs/bcm2/mer_models/mer_model_{today}_v{i}.h5'
hist_dest  = f'/scratch/gpfs/bcm2/mer_models/history_{today}_v{i}.csv'
model.save(model_dest)
history_df = pd.DataFrame(history.history)
history_df.to_csv(hist_dest)