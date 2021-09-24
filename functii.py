from typing_extensions import ParamSpec
import numpy as np
import copy


def sigmoid(z):

    # my own sigmoid function
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_parameters(n_x, n_y):
    np.random.seed(2)

    # initialize weight matrices randomly and biases with 0
    W1 = np.random.randn(n_y, n_x) * 0.001
    b1 = np.zeros((n_y, 1))

    parameters = {"W": W1, "b": b1}
    return parameters


def propagation(W, b, X, Y):

    # calculating the number of samples
    m = X.shape[1]
    # computing activation
    A = sigmoid(np.dot(W, X) + b)
    # computin costs
    logprobs = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), (1 - Y))
    cost = -np.sum(logprobs) * 1 / m
    cost = float(np.squeeze(cost))
    # computing grads
    dA = A - Y
    dw = 1 / m * np.dot(dA, X.T)
    db = 1 / m * np.sum(dA, axis=1, keepdims=True)
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    # actualizing the weight and bias matrices

    for i in range(num_iterations):

        grads, cost = propagation(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    # for the actualized parameters, calculate the outputs
    m = X.shape[1]
    Y_prediction = np.zeros((10, m))
    A = sigmoid(np.dot(w, X) + b)
    A = A.T
    for i in range(A.shape[0]):
        maxim_linie = max(A[i])
        for j in range(A.shape[1]):
            if A[i][j] == maxim_linie:
                Y_prediction[j][i] = 1.0
            else:
                Y_prediction[j][i] = 0.0

    return Y_prediction
