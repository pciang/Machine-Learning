import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, AveragePooling2D, Flatten, Activation, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

def relu_init(shape, name=None):
    value = np.random.normal(size=shape, loc=0.0, scale=0.1)
    return K.variable(value)

model = Sequential([
    Convolution2D(30, 3, 3, border_mode='valid', init=relu_init, input_shape=(1,28,28), dim_ordering='th'),
    Activation('relu'),
    AveragePooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
    Convolution2D(60, 4, 4, border_mode='valid', init=relu_init, input_shape=(1,28,28), dim_ordering='th'),
    Activation('relu'),
    AveragePooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
    Flatten(),
    Dense(600, init=relu_init),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])

sgd = SGD(lr=1e-2, momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

data = pd.read_csv('train.csv', dtype=np.float32).as_matrix()
M = data.shape[0]
X_train = data[:,1:]
X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
y_train = data[:,0]
X_train = np.reshape(X_train,(M,1,28,28))
y_train = np.repeat(np.arange(0,10)[None,:],M,axis=0) == np.repeat(y_train[:,None],10,axis=1)
y_train = y_train.astype(np.float32)

# Saving precious RAM?
del data

model.fit(X_train, y_train, batch_size=2000, nb_epoch=15)

ans = raw_input('continue? [y]/n: ')
if ans != '' and ans != 'y' and ans != 'Y':
    sys.exit(0)

X_test = pd.read_csv('test.csv', dtype=np.float32).as_matrix()
test_size = X_test.shape[0]
X_test = (X_test - X_mean) / X_std
X_test = np.reshape(X_test,(test_size,1,28,28))

y_hat = model.predict_classes(X_test, batch_size=2000)[:,None]
img_id=np.arange(1,28001)[:,None]
merged=np.concatenate((img_id,y_hat),axis=1)
np.savetxt('submission2.csv', merged, delimiter=',',fmt='%d', header='ImageId,Label', comments='')
