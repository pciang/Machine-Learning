import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, AveragePooling2D, Flatten, Activation, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras import backend as K

model = Sequential([
    Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1,28,28), dim_ordering='th'),
    Activation('relu'),
    AveragePooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
    Dropout(0.3),
    Convolution2D(64, 5, 5, border_mode='valid', input_shape=(1,28,28), dim_ordering='th'),
    Activation('relu'),
    AveragePooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
    Dropout(0.3),
    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.3),
    Dense(10),
    Activation('softmax'),
    ])

sgd = SGD(lr=1e-1, momentum=0.7)
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

model.fit(X_train, y_train, batch_size=100, nb_epoch=3)

ans = raw_input('continue? [y]/n: ')
if ans != '' and ans != 'y' and ans != 'Y':
    sys.exit(0)

X_test = pd.read_csv('test.csv', dtype=np.float32).as_matrix()
test_size = X_test.shape[0]
X_test = (X_test - X_mean) / X_std
X_test = np.reshape(X_test,(test_size,1,28,28))

y_hat = model.predict_classes(X_test, batch_size=100)[:,None]
img_id=np.arange(1,28001)[:,None]
merged=np.concatenate((img_id,y_hat),axis=1)
np.savetxt('submission2.csv', merged, delimiter=',',fmt='%d', header='ImageId,Label', comments='')
