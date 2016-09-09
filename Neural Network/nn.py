import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random

# Fixed
input_size = 2
bias_coeff = 1.

# Variable
hidden_size = 4
output_size = 2

def generate(n_sample=10):
    dataset = np.zeros((0, input_size + 1, ))
    for i in range(n_sample):
        while True:
            coord_x, coord_y = random.random(), random.random()

            # Classification
            kelas = 1

            radius = .33
            if (coord_x - .5)**2 + (coord_y - .5)**2 >= radius * radius:
                kelas = 2

            dataset = np.append(dataset, ((coord_x, coord_y, kelas, ),), axis=0)

            break

    return dataset

n_sample = 500
dataset = generate(n_sample)

X_nobias = dataset[:, :-1]
X_wbias = np.insert(X_nobias, 0, bias_coeff, axis=1)

y = np.zeros((n_sample, output_size, ))
for i in range(1, output_size + 1):
    y[:, i - 1] = dataset[:, -1] == i

W1 = np.random.uniform(-.5, .5, size=(input_size+1,hidden_size))
W2 = np.random.uniform(-.5, .5, size=(hidden_size+1,output_size))

def sigmoid(u):
    return 1. / (1. + np.exp(-u))

def sigmoid_prime(u):
    return sigmoid(u) * (1. - sigmoid(u))

def forward(X, params=None, itdout=False):
    '''
    Assume bias is included in X
    '''
    w1, w2 = W1, W2
    if params is not None:
        w1, w2 = params

    z2 = X.dot(w1)
    a2_nobias = sigmoid(z2)
    a2 = np.insert(a2_nobias, 0, bias_coeff, axis=1)
    z3 = a2.dot(w2)
    y_hat = sigmoid(z3)

    # If intermediate output is requested
    if itdout:
        return z2, a2_nobias, a2, z3, y_hat

    # Else
    return y_hat

def cost_function(X, y, params=None):
    '''
    Assume bias is included in X
    '''
    m = X.shape[0]
    y_hat = forward(X, params=params)
    return sum(sum(-y * np.log(y_hat) - (1. - y) * np.log(1. - y_hat))) / m

def backprop(X, y, max_iter=200, Alpha=0.03, detailed=False, tol=1e-6):
    '''
    Assume bias is included in X
    '''
    m = X.shape[0]
    w1, w2 = W1.copy(), W2.copy()
    
    previous_cost = cost_function(X, y, params=(w1,w2))
    
    for it in range(max_iter):
        z2, a2_nobias, a2, z3, y_hat = forward(X, params=(w1,w2), itdout=True)

        delta3 = y_hat - y
        delta2 = delta3.dot(w2[1:, :].T) * sigmoid_prime(z2)

        grad2 = a2.T.dot(delta3) / m
        grad1 = X.T.dot(delta2) / m

        w1[0, :] = grad1[0, :]
        w1[1:, :] -= Alpha * grad1[1:, :]

        w2[0, :] = grad2[0, :]
        w2[1:, :] -= Alpha * grad2[1:, :]

        current_cost = cost_function(X, y, params=(w1,w2))

        if detailed:
            print 'Iteration-%d: %.9f' % (it, current_cost)

        improvement = abs(previous_cost - current_cost)

        if improvement < tol:
            print 'Backprop stopped at iteration-%d: %.9f' % (it, current_cost)

            # Further improvement is too miniscule
            break

        previous_cost = current_cost

    print 'Backprop failed to converge: %.9f' % current_cost
    return w1, w2

W1, W2 = backprop(X_wbias, y, max_iter=10000, Alpha=1., detailed=False, tol=1e-9)

dx = .01
dy = .01
xs = np.arange(0., 1. + dx, dx)
ys = np.arange(0., 1. + dy, dy)
z = np.zeros((len(ys), len(xs)))

for i in range(len(ys)):
    for j in range(len(xs)):
        coord_x, coord_y = xs[j], ys[i]
        X_t = np.array([[bias_coeff, coord_x, coord_y]])
        y_t = forward(X_t)
        z[i][j] = y_t.argmax() + 1

xs = xs[None, :].repeat(len(ys), axis=0)
ys = ys[:, None].repeat(len(xs), axis=1)

# Plotting
plt.subplot(1, 2, 1)
plt.scatter(X_nobias[:, 0][dataset[:, -1] == 1], X_nobias[:, 1][dataset[:, -1] == 1], c='blue')
plt.scatter(X_nobias[:, 0][dataset[:, -1] == 2], X_nobias[:, 1][dataset[:, -1] == 2], c='red')
plt.title('Data: 1 - blue, 2 - red')
plt.axis([0., 1., 0., 1.])

plt.subplot(1, 2, 2)
plt.pcolormesh(xs, ys, z, cmap='bwr', vmin=1.0, vmax=output_size)
plt.title('Classification: 1 - blue, 2 - red')
plt.axis([0., 1., 0., 1.])
# plt.colorbar()
plt.show()
