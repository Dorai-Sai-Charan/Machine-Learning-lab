def bipolar_step(x):
    return 1 if x > 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

import numpy as np
import matplotlib.pyplot as plt

# Define initial weights and learning rate
initial_weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05

class Perceptron:
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.learning_rate = learning_rate

    def bipolar_step_function(self, x):
        return np.where(x >= 0, 1, -1)

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    def relu_function(self, x):
        return np.maximum(0, x)

    def predict(self, inputs, activation_function):
        weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if activation_function == 'bipolar_step':
            return self.bipolar_step_function(weighted_sum)
        elif activation_function == 'sigmoid':
            return self.sigmoid_function(weighted_sum)
        elif activation_function == 'relu':
            return self.relu_function(weighted_sum)

    def train(self, inputs, labels, activation_function, epochs=1000):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i], activation_function)
                error = labels[i] - prediction
                total_error += error**2
                self.weights[1:] += self.learning_rate * error * inputs[i]
                self.weights[0] += self.learning_rate * error

            average_error = total_error / len(inputs)
            errors.append(average_error)

            if average_error <= 0.002:
                break

        return errors, epoch + 1

# AND gate inputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND gate labels
labels = np.array([0, 0, 0, 1])

# Initialize Perceptron
perceptron = Perceptron(initial_weights, learning_rate)

activation_functions = ['bipolar_step', 'sigmoid', 'relu']
iterations = {}

for activation_function in activation_functions:
    perceptron = Perceptron(initial_weights, learning_rate)
    errors, num_iterations = perceptron.train(inputs, labels, activation_function)
    iterations[activation_function] = num_iterations

# Print number of iterations taken to converge for each activation function
for activation_function, num_iterations in iterations.items():
    print(f"Iterations taken to converge for {activation_function} activation function: {num_iterations}")

