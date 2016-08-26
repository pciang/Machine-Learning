import numpy as np
import matplotlib.pyplot as plt

X_nobias = np.array([
    [5, 1],
    [7, 3],
    [3, 2],
    [5, 4],
    [0, 0],
    [-1, -3],
    [-2, 3],
    [-3, 0],

    # Added data
    [-3, -3],
    [-3, -2],
    [-3, -1],
    [-3, 1],
    [-3, 2],
    [-3, 3],
    [-3, 4],
    [-3, 5],
    [-3, 6],
    [-2, -3],
    [-2, -2],
    [-2, -1],
    [-2, 0],
    [-2, 1],
    [-2, 2],
    [-2, 4],
    [-2, 5],
    [-1, -2],
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [-1, 2],
    [-1, 3],
    [-1, 4],
    
    [0, -3],
    [0, -2],
    [0, -1],
    [0, 1],
    [0, 2],
    [0, 3],

    [1, -3],
    [1, -2],
    [1, -1],
    [1, 1],
    [1, 2],

    [2, -3],
    [2, -2],
    [2, -1],
    [2, 1],

    [7, 7],
    [7, 6],
    [7, 5],
    [7, 4],
    [7, 2],
    [7, 1],
    [7, 0],
    [7, -1],
    [7, -2],
    [7, -3],

    [6, 7],
    [6, 6],
    [6, 5],
    [6, 4],
    [6, 3],
    [6, 2],
    [6, 1],
    [6, 0],
    [6, -1],
    [6, -2],

    [5, 7],
    [5, 6],
    [5, 5],
    [5, 3],
    [5, 2],
    [5, 0],
    [5, -1],

    [4, 7],
    [4, 6],
    [4, 5],
    [4, 4],
    [4, 3],
    [4, 2],
    [4, 1],
    [4, 0],

    [3, 7],
    [3, 6],
    [3, 5],
    [3, 4],
    [3, 3],
    [3, 1],
    ],

    dtype=np.float
    )

y = np.array([[1, 1, 1, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1,
               ]]).T

def train(alpha=0.3, num_iterations=1000):
    # Added bias
    X = np.insert(X_nobias, 0, 1, axis=1)

    params = np.random.normal(size=(3, 1))
    for it in range(num_iterations):
        y_hat = X.dot(params)
        y_hat[y_hat >= 0] = 1
        y_hat[y_hat < 0] = 0

        params = params - alpha * X.T.dot(y_hat - y)

    return params

def graph(params):
    temp_x = np.array(range(-3, 7))
    temp_y = (-params[0] - temp_x * params[1]) / params[2]
    plt.plot(temp_x, temp_y, c='c')
    plt.scatter(X_nobias[:, 0][y.flat == 1], X_nobias[:, 1][y.flat == 1], color='r')
    plt.scatter(X_nobias[:, 0][y.flat == 0], X_nobias[:, 1][y.flat == 0], color='b')
    plt.axis([-4, 8, -4, 8])
    plt.show()

graph(train())

X_normalized = X_nobias.copy()
X_normalized[:, 0] = (X_normalized[:, 0] - min(X_normalized[:, 0])) / (max(X_normalized[:, 0]) - min(X_normalized[:, 0])) * 2. - 1.
X_normalized[:, 1] = (X_normalized[:, 1] - min(X_normalized[:, 1])) / (max(X_normalized[:, 1]) - min(X_normalized[:, 1])) * 2. - 1.

def train_normalized(alpha=0.3, num_iterations=100):
    # Added bias
    X = np.insert(X_normalized, 0, 1, axis=1)

    params = np.random.normal(size=(3, 1))
    for it in range(num_iterations):
        y_hat = X.dot(params)
        y_hat[y_hat >= 0] = 1
        y_hat[y_hat < 0] = 0

        params = params - alpha * X.T.dot(y_hat - y)

    return params

def graph_normalized(params):
    temp_x = np.array(range(-2, 3))
    temp_y = (-params[0] - temp_x * params[1]) / params[2]
    plt.plot(temp_x, temp_y, c='c')
    plt.scatter(X_normalized[:, 0][y.flat == 1], X_normalized[:, 1][y.flat == 1], color='r')
    plt.scatter(X_normalized[:, 0][y.flat == 0], X_normalized[:, 1][y.flat == 0], color='b')
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.show()

graph_normalized(train_normalized())
