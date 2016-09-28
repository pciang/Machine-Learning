import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import lasagne
from lasagne.init import *
from lasagne.layers import *
from lasagne.updates import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.regularization import *

def load_dataset(val_ratio=0.2, test_ratio=0.2, shuffle=True, normalize=True):
    dataset_filename = 'train.csv'
    dataset_csv = pd.read_csv(dataset_filename)
    dataset_ndarr = dataset_csv.as_matrix()
    input_ndarr = dataset_ndarr[:, 1:].astype(float, copy=False)
    target_ndarr = dataset_ndarr[:, 0].astype(int, copy=False)
    dataset_size = dataset_ndarr.shape[0]
    train_ratio = 1. - val_ratio - test_ratio
    assert train_ratio > 0., 'wtf'
    train_size = int(math.ceil(train_ratio * dataset_size))
    val_size = int(math.floor(val_ratio * dataset_size))
    test_size = int(math.floor(test_ratio * dataset_size))
    if shuffle:
        shuffled_idx = np.arange(dataset_size)
        np.random.shuffle(shuffled_idx)
        dataset_ndarr = dataset_ndarr[shuffled_idx]
    if normalize:
        # input_mean = input_ndarr.mean(axis=0)
        # input_std = input_ndarr.std(axis=0)
        # input_std[input_std == 0.] = 1. # Prevent zero division error
        # input_ndarr = (input_ndarr - input_mean) / input_std
        input_ndarr /= 255. # 0-1 normalization
    y_all = np.zeros((dataset_size, 10))
    y_all[np.arange(dataset_size), target_ndarr] = 1.
    X_train = input_ndarr[:train_size].reshape((train_size, 1, 28, 28))
    y_train = y_all[:train_size]
    X_val, y_val, X_test, y_test = None, None, None, None
    if val_size > 0:
        X_val = input_ndarr[train_size:train_size + val_size].reshape((val_size, 1, 28, 28))
        y_val = y_all[train_size:train_size + val_size]
    if test_size > 0:
        X_test = input_ndarr[-test_size:].reshape((test_size, 1, 28, 28))
        y_test = y_all[-test_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_test(normalize=True):
    testset_filename = 'test.csv'
    testset_csv = pd.read_csv(testset_filename)
    X_test = testset_csv.as_matrix().astype(float, copy=False)
    test_size = X_test.shape[0]
    if normalize:
        X_test /= 255.
    X_test = X_test.reshape((test_size, 1, 28, 28))
    return X_test

def mini_batches(X, y, size=500):
    m = X.shape[0]
    assert m == y.shape[0], 'wth?'
    for start_idx in range(0, m, size):
        yield X[start_idx:start_idx+size], y[start_idx:start_idx+size]

if __name__ == '__main__':
    max_iter = 250
    Alpha = 1e-2 # Learning rate
    mini_batch_size = 500
    print('Import done! Loading dataset...')
    X_train, y_train, X_val, y_val, _, _ = load_dataset(val_ratio=0.3, test_ratio=0.)
    print('Dataset loaded! Building network...')
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')
    input_layer = InputLayer((None,1,28,28), input_var=input_var)
    conv_layer = Conv2DLayer(
        input_layer,
        num_filters=10,
        filter_size=(5,5),
        nonlinearity=linear,
        W=Uniform(range=(-0.1,0.1)),
        b=Uniform(range=(-0.1,0.1)),
        )
    pool_layer = MaxPool2DLayer(
        conv_layer,
        pool_size=(2,2),
        stride=None,
        )
    output_layer = DenseLayer(
        pool_layer,
        num_units=10,
        W=Uniform(range=(-0.1,0.1)),
        b=Uniform(range=(-0.1,0.1)),
        nonlinearity=sigmoid
        )
    conv_network = output_layer
    print('Model built! Initiate trainig...')
    prediction = get_output(conv_network)
    params = get_all_params(conv_network, trainable=True)
    loss = binary_crossentropy(prediction, target_var).mean()
    updates = sgd(loss, params, learning_rate=Alpha)
    accuracy = T.mean(T.eq(T.argmax(prediction,axis=1), T.argmax(target_var,axis=1)))
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [loss, accuracy])
    predict_fn = theano.function([input_var], prediction)
    print('Training...')
    training_start_time = time.time()
    for epoch in range(max_iter):
        epoch_start_time = time.time()
        train_err = 0.
        train_nbatch = 0.
        for X, y in mini_batches(X_train, y_train, size=mini_batch_size):
            w = float(X.shape[0]) / mini_batch_size
            train_err += w * train_fn(X, y)
            train_nbatch += w
        val_err = 0.
        val_acc = 0.
        val_nbatch = 0.
        for X, y in mini_batches(X_val, y_val, size=mini_batch_size):
            w = float(X.shape[0]) / mini_batch_size
            err, acc = val_fn(X, y)
            val_err += w * err
            val_acc += w * acc
            val_nbatch += w
        print('Epoch %d took %.3fs' % (epoch, time.time() - epoch_start_time))
        print('    training loss: %.9f' % (train_err / train_nbatch))
        print('    cv loss: %.9f' % (val_err / val_nbatch))
        print('    cv acc: %.9f' % (val_acc / val_nbatch))
    print('Total training time: %.9f' % (time.time() - training_start_time))
    print('Loading test...')
    X_test = load_test()
    test_size = X_test.shape[0]
    result = np.zeros((test_size,2))
    result[:, 0] = np.arange(test_size) + 1
    print('Testset loaded! Predicting...')
    for start_idx in range(0, test_size, 500):
        predicted = predict_fn(X_test[start_idx:start_idx+500]).argmax(axis=1, )
        result[start_idx:start_idx+500,1] = predicted
    np.savetxt(
        'submission.csv',
        result,
        fmt='%d',
        delimiter=',',
        header='ImageId,Label',
        comments=''
        )
    print('Finished predicting!')
