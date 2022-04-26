import random
import math
import csv


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def randomWeights(n):   # creates list of n size
    random.seed()
    return [round(random.random(), 4) for i in range(n)]


def read_csv(filename):
    file = open(filename)
    reader = csv.reader(file)
    next(reader)    # ignore header
    data = list()
    for line in reader:
        data.append(line)
    return data


def input_outputs(array):  # separates inputs & specie(output) from given dataset
    inputs = list()
    outputs = list()
    index = 0
    for row in array:
        inputs.append(row)
        outputs.append(inputs[index].pop())
        index += 1
    return [inputs, outputs]


def perceptron(inputs, weights):
    total_sum = 0.0
    for i in range(len(inputs) + 1):
        if i == 0:
            total_sum += 1 * weights[i]
        else:
            total_sum += float(inputs[i - 1]) * weights[i]
    return sigmoid(total_sum)


def NeuralNetwork(inputs, layer_1, layer_2):    # returns predicted value for inputs
    # hidden layer - 1
    outputs_1 = list()
    for i in range(layer_1):
        outputs_1.append(perceptron(inputs, randomWeights(len(inputs) + 1)))

    # hidden layer - 2
    outputs_2 = list()
    for i in range(layer_2):
        outputs_2.append(perceptron(outputs_1, randomWeights(len(outputs_1) + 1)))

    return 1 + outputs_2.index(max(outputs_2))      # selects perceptron with max result


def Accuracy(predicted, actual):     # returns % of matches
    matches = 0
    total = 0

    for i in range(len(actual)):
        if predicted[i] == actual[i]:
            matches += 1
        total += 1

    return 100 * (matches / total)

