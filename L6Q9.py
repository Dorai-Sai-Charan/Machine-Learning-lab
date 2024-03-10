from calendar import EPOCH
import numpy as np

def step_function(x):
  """Defines the step activation function."""
  return int(x > 0)  # Threshold at 0, output 1 if greater than 0, else 0

def train_perceptron(inputs, targets, learning_rate, epochs=1000):
  """Trains a single-layer perceptron with backpropagation and step activation.

  Args:
      inputs: A NumPy array of training inputs (each row represents an input sample).
      targets: A NumPy array of desired outputs for the corresponding inputs.
      learning_rate: The learning rate for weight updates.
      epochs: The maximum number of epochs to train for (default: 1000).

  Returns:
      A tuple containing the final weights and the convergence epoch (if achieved).
  """

  # Initialize weights with provided values
  w0 = 10
  w1 = 0.2
  w2 = -0.75
  bias = 0  # Can be added for bias term

  for epoch in range(epochs):
    total_error = 0
    for i in range(len(inputs)):
      # Forward pass
      weighted_sum = w0 + np.dot(inputs[i], [w1, w2]) + bias  # Include bias if used

      # Apply step activation
      predicted_output = step_function(weighted_sum)

      # Calculate error
      error = targets[i] - predicted_output

      # Backpropagation (simplified for step function)
      # Update weights based on error and input values
      w0 += learning_rate * error
      w1 += learning_rate * error * inputs[i][0]
      w2 += learning_rate * error * inputs[i][1]

      total_error += error**2

    # Check for convergence
    average_error = total_error / len(inputs)
    if average_error <= 0.002:
      return w0, w1, w2, epoch + 1  # Return weights and epoch of convergence

  # Return weights if convergence not reached
  return w0, w1, w2, epochs

# Training data (XOR gate)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Train the perceptron
learning_rate = 0.05
w0, w1, w2, converged_epoch = train_perceptron(inputs, targets, learning_rate)

# Print results
if converged_epoch < 1000:
  print("Converged in", converged_epoch, "epochs.")
  print("Weights:")
  print("w0:", w0)
  print("w1:", w1)
  print("w2:", w2)
else:
  print("Convergence not reached within", EPOCH, "epochs.")
  """In essence, the single-layer perceptron with a step activation function simply lacks the necessary
    complexity to represent the non-linear decision boundary required for perfect XOR classification."""
