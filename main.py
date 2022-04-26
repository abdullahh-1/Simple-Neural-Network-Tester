import neuralnetwork as nn

if __name__ == '__main__':
    inputs, actual = nn.input_outputs(nn.read_csv("Neural Network/iris dataset.csv"))
    input_count = len(inputs[0])
    data_count = len(actual)
    layer_1 = 16    # 16 perceptron in hidden layer 1
    layer_2 = 3     # 03 perceptron in hidden layer 2
    predicted = list()

    for i in range(data_count):
        predicted.append(nn.NeuralNetwork(inputs[i], layer_1, layer_2))

    print(nn.Accuracy(predicted, actual))
