# Handwritten-Digit-Recognition
## Assignment 1

Implement a model, based on the perceptron algorithm, that will be able to classify hand written digits. For training, use the mnist dataset.

https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz

The classification algorithm must use 10 perceptrons. Each of the 10 perceptrons will be trained to
distinguish elements that represent a specific digit from the rest of the elements in the dataset. For
example, the 0 perceptron will be trained to classify 0 from the digits 1,2,3,4,5,6,7,8,9
When you want to classify a digit from the test set, you will get the output of each of the 10 perceptrons.
The value will be given by the perceptron number with the output 1 or by the perceptron number with
the greatest output before the activation function.

## Assignment 2

Implement a classifier based on a fully connected feed forward neural network (multi-layer perceptron),
for hand written digits. For training, use the same dataset as in the first assignment.
The classification algorithm must use a neural network of 3 layers(input: 784 neurons, hidden: 100,
output: 10). The neurons from the hidden layer will use the sigmoid activation function. The neurons
from the last layer can also use sigmoid, but softmax is recommended. The output of the neural network
will be given by the neuron’s number form the final layer with the greatest output value.
