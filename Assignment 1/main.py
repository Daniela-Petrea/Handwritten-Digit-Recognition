import gzip
import pickle
import numpy as np
import numpy.random


def activation(n):
    if n > 0:
        return 1
    else:
        return 0


def shuffle(a, b):
    random_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(random_state)
    numpy.random.shuffle(b)


def get_batches(n, dataset):
    shuffle(dataset[0], dataset[1])
    batch_len = len(dataset[0]) // n
    batches = []
    for i in range(n):
        batch = []
        for j in range(i * batch_len, (i + 1) * batch_len):
            batch.append([dataset[0][j], dataset[1][j]])
        batches.append(batch)
    return batches


def create_list_for_target(size, target):
    list = [0] * size
    list[target] = 1
    return list


def learning(dataset, weights, bias, epochs_length, learning_rate, nr_batches):
    for i in range(epochs_length):
        batches = get_batches(nr_batches, dataset)
        batch_len = len(dataset[0]) // nr_batches
        for j in range(nr_batches):
            delta_weight, delta_bias = 0, 0
            for k in range(batch_len):
                y = []
                x = batches[j][k][0]
                t = batches[j][k][1]
                z = np.dot(weights, x) + bias.T
                for l in z.T:
                    y.append(activation(l))
                output = create_list_for_target(10, t)
                x = x.reshape(1, len(dataset[0][1]))
                delta_weight += np.dot((np.subtract(output, y)).reshape(10, 1), x) * learning_rate
                delta_bias += np.subtract(output, y).reshape(10, 1) * learning_rate
            weights += delta_weight
            bias += delta_bias
    return weights, bias


def train(dataset, epochs_len, learning_rate, nr_batches):
    print("Neural Network Training...")
    all_weights = 2*np.random.rand(10, len(dataset[0][1]))-1
    all_biases = 2*np.random.rand(10, 1)-1
    weight, bias = learning(dataset, all_weights, all_biases, epochs_len, learning_rate, nr_batches)
    return weight, bias


def validation_data(dataset, weights, bias):
    misclassified = 0
    y = [0] * 10
    for i in range(len(dataset[0])):
        list_of_z = []
        x = dataset[0][i]
        t = dataset[1][i]
        for j in range(10):
            z = np.dot(weights[j], x) + bias[j]
            list_of_z.append(z)
            y[j] = activation(z)
        if y.count(1) > 1 or y.count(1) == 0:
            max_value = max(list_of_z)
            index = list_of_z.index(max_value)
        else:
            index = y.index(1)
        output = index
        if t != output:
            misclassified += 1
    error = misclassified / len(dataset[0])
    accuracy = (1 - error) * 100
    return accuracy


if __name__ == '__main__':
    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        train_test, valid_set, test_set = pickle.load(fd, encoding='latin')
    all_weights, bias = train(train_test, 10, 0.05, 64)
    print("Weights and Bias Learned!")
    print("Neural Network Validation...")
    print(f"Train Test Accuracy -> {validation_data(train_test, all_weights, bias)}%")
    print(f"Validation Set Accuracy -> {validation_data(valid_set, all_weights, bias)}%")
    print(f"Test Set Accuracy -> {validation_data(test_set, all_weights, bias)}%")
