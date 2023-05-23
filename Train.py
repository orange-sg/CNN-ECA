import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
from CNNECA import CNN_ECA

lat_res = 46
lon_res = 71
varn = 8
object_n = 464

def load_dataset():
    X = np.load('data\\era_plev\\full\\' + 'X_final.npy')
    Y = np.load('data\\full\\' + 'wind.npy')
    return X, Y

def normalize(data):
    data = data - data.mean()
    data = data / data.std()
    return data

def set_data(X, Y):
    X_normalized = np.zeros((varn, np.max(X.shape), lat_res, lon_res))
    for i in range(varn):
        X_normalized[i,] = normalize(X[i,])
    X = X_normalized.transpose(1, 2, 3, 0)
    Y
    return X, Y

def data_generator(X, Y):
    train_months = 300
    test_months = 360
    train_x, train_y = X[:train_months], Y[:train_months]
    test_x, test_y = X[train_months:test_months], Y[train_months:test_months]
    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)
    return train_x, train_y, test_x, test_y

X, Y = load_dataset()
X, Y = set_data(X, Y)

train_x, train_y, test_x, test_y = data_generator(X,Y)
train_x = train_x.reshape((train_x.shape[0],train_x.shape[3],train_x.shape[1],train_x.shape[2]))
test_x = test_x.reshape((test_x.shape[0],test_x.shape[3],test_x.shape[1],test_x.shape[2]))

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

model = CNN_ECA()
adam = tf.keras.optimizers.Adam(lr=0.003)
model.compile(optimizer=adam,
                  loss=root_mean_squared_error,
                  metrics=["mae","acc"])
checkpoint_save_path = ".\\checkpoint\\stock_era5_plev\\"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f".\\Graphs\\era5_plev_Graph\\", histogram_freq=0,
                                                 write_graph=True, write_images=False)
termnan = tf.keras.callbacks.TerminateOnNaN()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=15,
                                                     min_delta=0.005, min_lr=0.000004,
                                                     verbose=1)
callbacks_list = [checkpoint, tensorboard, reduce_lr, termnan]
history = model.fit(train_x,train_y, batch_size=20, epochs=1000, validation_data=(test_x, test_y), callbacks=callbacks_list)
model.summary()
loss = history.history['loss']
val_loss = history.history['val_loss']