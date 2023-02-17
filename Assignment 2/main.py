import gzip
import math
import pickle

import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid_derivative(a):
    s = 1 / (1 + np.exp(-a))
    return s * (1 - s)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def cost_function(a2, y, batch_size):
    cost = -(1 / batch_size) * np.sum(y * np.log(a2))

    return cost


def activation_function_batch(matrix, activation_function):
    list_with_activation = []
    for row in matrix:
        list_with_activation.append(activation_function(row))
    return np.row_stack(list_with_activation)


def activation(input):
    predictions_list = []
    for row in input:
        predictions_list.append(np.argmax(row))
    return predictions_list


def accuracy(preds, true):
    return np.sum(preds == true) / true.size * 100


def initialize_parameters(n_input, n_output):
    weights = np.random.normal(0, 1 / math.sqrt(n_input), size=(n_input, n_output))
    bias = np.random.rand(1, n_output)
    return weights, bias


def shuffle(X_train, Y_train):
    x = np.array(X_train)
    y = np.array(Y_train)
    r_indexes = np.arange(len(x))
    np.random.shuffle(r_indexes)
    x = x[r_indexes]
    y = y[r_indexes]
    return x, y


def model(X_train, Y_train, learning_rate, epochs, batch_size):
    w1, b1 = initialize_parameters(784, 100)
    w2, b2 = initialize_parameters(100, 10)
    for epoch in range(epochs):
        shuffle(X_train, Y_train)
        number_of_batches = len(X_train) // batch_size
        expected_output = np.zeros((batch_size, 10))
        predicted_output = np.zeros((batch_size, 10))
        for i in range(number_of_batches):
            x = X_train[(i * batch_size):(i + 1) * batch_size]
            y = Y_train[(i * batch_size):(i + 1) * batch_size]
            z1 = np.dot(x, w1) + b1
            a1 = activation_function_batch(z1, sigmoid)
            z2 = np.dot(a1, w2) + b2
            a2 = activation_function_batch(z2, softmax)
            expected_output = np.zeros((batch_size, 10))
            predicted_output = np.zeros((batch_size, 10))
            activate = activation(a2)
            for j in range(batch_size):
                expected_output[j, y[j]] = 1
                predicted_output[j, activate[j]] = 1
            dz2 = (a2 - expected_output)
            dw2 = (1 / batch_size) * np.dot(a1.T, dz2)
            db2 = (1 / batch_size) * np.sum(dz2, axis=0, keepdims=True)
            dz1 = np.dot(dz2, w2.T) * sigmoid_derivative(a1)
            dw1 = (1 / batch_size) * np.dot(x.T, dz1)
            db1 = (1 / batch_size) * np.sum(dz1, axis=0, keepdims=True)
            w1 = w1 - (learning_rate * dw1)
            b1 = b1 - (learning_rate * db1)
            w2 = w2 - (learning_rate * dw2)
            b2 = b2 - (learning_rate * db2)

            cross_entropy_cost = cost_function(a2, expected_output, batch_size)
            L2_regularization_cost = (np.sum(np.square(w1)) + np.sum(np.square(w2))) * (
                    learning_rate / (2 * batch_size))
            cost = cross_entropy_cost + L2_regularization_cost
            #print("Cost for each batch is",cost)

        print("Epoch ", epoch, ": accuracy of Train Dataset", accuracy(predicted_output, expected_output), "%")
    return w1, w2, b1, b2


def score(X, Y, w1, w2, b1, b2, batch_size):
    expected_output = np.zeros((batch_size, 10))
    predicted_output = np.zeros((batch_size, 10))
    shuffle(X, Y)
    number_of_batches = len(X) // batch_size
    for i in range(number_of_batches):
        x = X[(i * batch_size):(i + 1) * batch_size]
        y = Y[(i * batch_size):(i + 1) * batch_size]
        z1 = np.dot(x, w1) + b1
        a1 = activation_function_batch(z1, sigmoid)
        z2 = np.dot(a1, w2) + b2
        a2 = activation_function_batch(z2, softmax)
        expected_output = np.zeros((batch_size, 10))
        predicted_output = np.zeros((batch_size, 10))
        activate = activation(a2)
        for j in range(batch_size):
            expected_output[j, y[j]] = 1
            predicted_output[j, activate[j]] = 1
    return accuracy(expected_output, predicted_output)


if __name__ == '__main__':
    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
    X_train, Y_train = train_set[0], train_set[1]
    X_test, Y_test = test_set[0], test_set[1]
    X_valid, Y_valid = valid_set[0], valid_set[1]
    print("shape of X_train :", X_train.shape)
    print("shape of Y_train :", Y_train.shape)
    print("shape of X_test :", X_test.shape)
    print("shape of Y_test :", Y_test.shape)
    w1, w2, b1, b2 = model(X_train, Y_train, learning_rate=0.01, epochs=10, batch_size=128)
    print("Accuracy of Train Dataset: ", score(X_train, Y_train, w1, w2, b1, b2, 128), "%")
    print("Accuracy of Test Dataset: ", score(X_test, Y_test, w1, w2, b1, b2, 128), "%")
    print("Accuracy of Valid Dataset: ", score(X_valid, Y_valid, w1, w2, b1, b2, 128), "%")
