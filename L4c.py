import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# Load the dataset
dataset = pd.read_excel("C:\\OS\\Machine learning\\Machine Learning Lab\\LAB4-20.02.24\\training_mathbert 2.xlsx")

# Select two feature vectors for distance calculation
feature_1 = 'embed_0'
feature_2 = 'embed_1'

# Extract the values of the selected features
X = dataset[[feature_1, feature_2]].values

# Divide the dataset into train and test sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Calculate Minkowski distance for different values of r
r_values = range(1, 11)
train_distances = []
test_distances = []

for r in r_values:
    train_distance_r = minkowski(X_train[:, 0], X_train[:, 1], p=r)
    test_distance_r = minkowski(X_test[:, 0], X_test[:, 1], p=r)
    train_distances.append(train_distance_r)
    test_distances.append(test_distance_r)

# Plot the distance versus r
plt.figure(figsize=(10, 6))
plt.plot(r_values, train_distances, marker='o', linestyle='-', label='Train Dataset')
plt.plot(r_values, test_distances, marker='x', linestyle='-', label='Test Dataset')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot to visualize train and test datasets
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], color='blue', label='Train Dataset')
plt.scatter(X_test[:, 0], X_test[:, 1], color='red', label='Test Dataset')
plt.title('Scatter Plot of Train and Test Datasets')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.legend()
plt.grid(True)
plt.show()
