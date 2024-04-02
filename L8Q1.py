import numpy as np
from collections import Counter

def calculate_entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_information_gain(X, y, feature_index):
    total_entropy = calculate_entropy(y)
    
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    
    # Calculating weighted entropy for each value of the feature
    weighted_entropy = np.sum([(counts[i] / len(X)) * calculate_entropy(y[X[:, feature_index] == values[i]]) for i in range(len(values))])
    
    # Calculating information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_root_node_value(X, y):
    max_information_gain = -1
    root_feature_value = None
    
    # Iterate over each feature to find the one with maximum information gain
    for i in range(X.shape[1]):
        information_gain = calculate_information_gain(X, y, i)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            root_feature_value = X[0, i]  # Set root feature value to the value of the first instance
    
    return root_feature_value


#dataset
X = np.array([
    [12, 'A', 'X', 1],
    [7, 'C', 'Y', 0],
    [9, 'A', 'Y', 1],
    [4, 'B', 'X', 1],
    [5, 'B', 'X', 0],
    [6, 'A', 'Y', 0],
    [3, 'C', 'Z', 1],
    [2, 'A', 'Z', 0],
    [1, 'B', 'X', 1],
    [10, 'C', 'Z', 1]
])
y = np.array([2, 7, 0, 6, 3, 4, 1, 8, 5, 9])

# Finding the root node
root_feature_value = find_root_node_value(X, y)
print("Root feature value:", root_feature_value)
