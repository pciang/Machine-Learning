import numpy as np
from scipy import optimize
import time
import gc
from PIL import Image
import random
from bisect import bisect_right

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z) * (1. - sigmoid(z))

# does not include bias
input_layer_size = 64
hidden_layer_size = 16
output_layer_size = 64

W1 = np.random.uniform(-1., 1., (input_layer_size + 1, hidden_layer_size))
W2 = np.random.uniform(-1., 1., (hidden_layer_size + 1, output_layer_size))

Lambda = 0.01

def forward(X):
    '''
    Assumption: X already include bia
    '''
    z2 = np.dot(X, W1)

    # for 2d array
    if len(X.shape) == 2:
        a2 = np.insert(sigmoid(z2), 0, 1., axis = 1)

    # for 1d array
    else:
        a2 = np.insert(sigmoid(z2), 0, 1)
    
    z3 = np.dot(a2, W2)
    
    return sigmoid(z3)

def setParams(params):
    global W1, W2
    W1_end = (input_layer_size + 1) * hidden_layer_size
    W1 = np.reshape(params[0:W1_end], (input_layer_size + 1, hidden_layer_size))

    W2_end = W1_end + (hidden_layer_size + 1) * output_layer_size
    W2 = np.reshape(params[W1_end:W2_end], (hidden_layer_size + 1, output_layer_size))

def costFunction(X, y):
    yHat = forward(X)
    m = X.shape[0]
    return sum(sum(-y * np.log(yHat) - (1. - y) * np.log(1. - yHat))) / m \
            + Lambda * (sum(sum(W1[1:, :] ** 2)) + sum(sum(W2[1:, :] ** 2))) / (2. * m)

def flattenParams():
    return np.concatenate((W1.ravel(), W2.ravel()) )

def backprop(X, y):
    '''
    Assumption: X already include bias
    '''
    grad1 = np.zeros(W1.shape)
    grad2 = np.zeros(W2.shape)

    m = X.shape[0]
    z2 = np.dot(X, W1)
    a2 = np.insert(sigmoid(z2), 0, 1., axis = 1)
    z3 = np.dot(a2, W2)
    a3 = sigmoid(z3)

    delta3 = a3 - y
    delta2 = np.dot(delta3, W2[1:, :].T) * sigmoidPrime(z2)
    grad2 += np.dot(a2.T, delta3)
    grad1 += np.dot(X.T, delta2)

    grad1 /= m
    grad2 /= m

    grad1[1:, :] += Lambda * grad1[1:, :] / m
    grad2[1:, :] += Lambda * grad2[1:, :] / m

    return grad1, grad2

def costFunctionWrapper(params):
    setParams(params)
    cost = costFunction(X, y)

    grad1, grad2 = backprop(X, y)
    grads = np.concatenate((grad1.ravel(), grad2.ravel()) )

    return cost, grads

def checkGradient(X, y):
    paramsInitial = flattenParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        perturb[p] = e
        setParams(paramsInitial + perturb)
        loss1 = costFunction(X, y)

        setParams(paramsInitial - perturb)
        loss2 = costFunction(X, y)

        numgrad[p] = (loss1 - loss2) / (2. * e)

        perturb[p] = 0

    setParams(paramsInitial)
    return numgrad

def train(X, y):
    '''
    Assumption: X already include bias
    '''
    params0 = flattenParams()
    options = {'maxiter': 200, 'disp': True}
    _res = optimize.minimize(costFunctionWrapper, params0, jac = True, method = 'L-BFGS-B', \
                              options = options)

    return _res


filename = 'learn.jpg'
img = Image.open(filename)
print 'Opening file "{0}"'.format(filename)

colors = img.load()
width, height = img.size

N_sample = 300
X = np.empty((N_sample, input_layer_size + 1))
y = np.empty((N_sample, output_layer_size))

for s in range(N_sample):
    r = random.randint(0, height - 9)
    c = random.randint(0, width - 9)
    
    X[s, 0] = 1.
    idx = 0
    choice = random.randint(0, 2)
    for i in range(r, r + 8):
        for j in range(c, c + 8):
            idx += 1
            X[s, idx] = y[s, idx - 1] = colors[j, i][choice] / 255.

print train(X, y)

hidden_output_bitsize = 4
Range = [-1. + 2. * x / (1 << hidden_output_bitsize) \
         for x in range((1 << hidden_output_bitsize) + 1)]

def compress(filename):
    img = Image.open(filename)
    print 'Image "{}" loaded!'.format(filename)

    colors = img.load()
    width, height = img.size

    output = np.empty((height, width, 3))

    dataX = np.empty(input_layer_size + 1)
    dataX[0] = 1.
    for color in range(3):
        for r1 in range(0, height / 8):
            for c1 in range(0, width / 8):
                idx = 0
                for i in range(r1 * 8, r1 * 8 + 8):
                    for j in range(c1 * 8, c1 * 8 + 8):
                        idx += 1
                        dataX[idx] = colors[j, i][color] / 255.

                hidden_output = sigmoid(np.dot(dataX, W1))

                # compressing
                for i in range(len(hidden_output)):
                    idx = bisect_right(Range, hidden_output[i])
                    
                    if idx >= len(Range):
                        idx -= 1

                    hidden_output[i] = (Range[idx - 1] + Range[idx]) / 2.

                hidden_output = np.insert(hidden_output, 0, 1, axis = 0)
                y = sigmoid(np.dot(hidden_output, W2))

                idx = 0
                for i in range(r1 * 8, r1 * 8 + 8):
                    for j in range(c1 * 8, c1 * 8 + 8):
                        output[i, j, color] = np.floor(255. * y[idx])
                        idx += 1

    compressed_img = Image.fromarray(output.astype(np.uint8), 'RGB')
    compressed_img.show()

    return compressed_img
