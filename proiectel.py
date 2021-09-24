import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import functii as fnc

# Loading the dataset into a variable and separate the inputs from outputs
mnist = loadmat("mnist-original.mat")
mnist_data = mnist["data"].T
mnist_data_pt_calcule = mnist["data"]
mnist_label = mnist["label"][0].reshape(70000, 1)

matrice_raspuns = np.zeros((mnist_label.shape[0], 10))
for i in range(len(mnist_label)):
    matrice_raspuns[i][int(mnist_label[i][0])] = 1

# Splitting the dataset into the trainset and testset
X_train, X_test, Y_train, Y_test = train_test_split(
    mnist_data, matrice_raspuns, random_state=4
)

# Transposing the matrices, to put each input/output in a column
X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T

# Centering and standardizing the dataset
X_train = X_train / 255
X_test = X_test / 255

# Initialize parameters
parameters = fnc.initialize_parameters(X_train.shape[0], Y_train.shape[0])
w = parameters["W"]
b = parameters["b"]

# Optimizing parameters
params, grads, costs = fnc.optimize(w, b, X_train, Y_train, 5001, 0.009, True)

w = params["w"]
b = params["b"]

# Calculating y for the train and test set
Y_prediction_train = fnc.predict(w, b, X_train)
Y_prediction_test = fnc.predict(w, b, X_test)

print(
    "train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    )
)
print(
    "test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    )
)
