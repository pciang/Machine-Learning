import numpy as np
from scipy import optimize
import time
import gc
from PIL import Image

# does not include bias
input_layer_size = 400
hidden_layer_size = 100
output_layer_size = 10

W1 = np.random.uniform(-1., 1., (input_layer_size + 1, hidden_layer_size))
W2 = np.random.uniform(-1., 1., (hidden_layer_size + 1, output_layer_size))

Lambda = 0.01

start_time = time.time()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X):
    '''
    Assumption: X already include bias
    '''
    z2 = np.dot(X, W1)
    a2 = np.insert(sigmoid(z2), 0, 1, axis = 1)
    z3 = np.dot(a2, W2)
    yHat = sigmoid(z3)
    
    return yHat
    
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def setParams(params):
    global W1, W2
    W1_start = 0
    W1_end = (input_layer_size + 1) * hidden_layer_size
    W1 = np.reshape(params[W1_start : W1_end], (input_layer_size + 1, hidden_layer_size))
    
    W2_end = W1_end + (hidden_layer_size + 1) * output_layer_size
    W2 = np.reshape(params[W1_end : W2_end], (hidden_layer_size + 1, output_layer_size))

    return

def costFunction(X, y):
    yHat = forward(X)
    m = X.shape[0]
    return sum(sum(-y * np.log(yHat) - (1 - y) * np.log(1 - yHat))) / m \
            + Lambda * (sum(sum(W1[1:, :] ** 2)) + sum(sum(W2[1:, :] ** 2))) / (2 * m)

def backprop(X, y):
    '''
    Assumption: X already include bias
    '''
    grad1 = np.zeros(W1.shape, dtype = float)
    grad2 = np.zeros(W2.shape, dtype = float)

    a1 = X
    m = a1.shape[0]
    z2 = np.dot(X, W1)
    a2_nobias = sigmoid(z2)
    a2 = np.insert(a2_nobias, 0, 1, axis = 1)
    z3 = np.dot(a2, W2)
    a3 = sigmoid(z3)

    delta3 = a3 - y

    delta2 = np.dot(delta3, W2[1:, :].T) * sigmoidGradient(z2)
    grad2 += np.dot(a2.T, delta3)
    grad1 += np.dot(a1.T, delta2)

    grad1 /= m
    grad2 /= m

    grad1[1:, :] += Lambda * grad1[1:, :] / m
    grad2[1:, :] += Lambda * grad2[1:, :] / m

    return grad1, grad2

def flattenParams():
    return np.concatenate((W1.ravel(), W2.ravel()) )

def costFunctionWrapper(params, X, y):
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
        loss2 = costFunction(X, y)

        setParams(paramsInitial - perturb)
        loss1 = costFunction(X, y)

        numgrad[p] = (loss2 - loss1) / (2. * e)

        perturb[p] = 0

    setParams(paramsInitial)
    return numgrad

def __cb(params):
    global cost_vs_iteration, start_time
    setParams(params)
    cost_vs_iteration.append(costFunction(X, y))
    print('Iteration-{}: {}, takes {}s'.format(len(cost_vs_iteration), cost_vs_iteration[-1], time.time() - start_time))

    gc.collect()

    start_time = time.time()

    return

cost_vs_iteration = []
def train(X, y):
    '''
    Assumption: X already include bias
    '''
    params0 = flattenParams()
    options = {'maxiter': 100, 'disp': True}
    _res = optimize.minimize(costFunctionWrapper, params0, jac = True, method = 'L-BFGS-B', \
                             args = (X, y), options = options, callback = __cb)

    setParams(_res.x)
    return _res

dataX = np.random.rand(1, input_layer_size)
y = np.array([np.arange(output_layer_size)], dtype = float)

digit = 0

f_in = open('data.txt', 'r')
for line in f_in.read().splitlines():
    if len(line) == 1:
        digit = int(line)
        continue

    datum = []
    for num_str in line.split(','):
        num = float(num_str)
        datum.append(num)
    dataX = np.concatenate((dataX, np.array([datum])), axis = 0)
    y = np.concatenate((y, np.array([np.arange(output_layer_size)]) == digit) \
             , axis = 0)

dataX = dataX[1:, :]
y = y[1:, :]

X = np.insert(dataX, 0, 1, axis = 1)

print(train(X, y))

for digit in range(10):
    print '\nExpected digit: {0}'.format(digit)
    for i in range(100):
        filename = 'data/test{0}_{1}.png'.format(digit, i)
        try:
            img = Image.open(filename)
            print 'Opening file "{0}"'.format(filename)
            pix = img.load()
            outputArr = []
            for r in range(img.size[1]):
                for c in range(img.size[0]):
                    val = (255.0 - pix[c, r][0]) / 255.0
                    outputArr.append(val)

            t = np.insert(np.array([outputArr]), 0, 1, axis = 1)

            result = forward(t)
            print 'Predicted: {0}'.format(np.argmax(result, axis = 1)[0])
            
        except IOError:
            break
