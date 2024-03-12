import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Load the dataset
dataset = pd.read_excel("C:\\OS\\Machine learning\\Machine Learning Lab\\LAB4-20.02.24\\training_mathbert 2.xlsx")

# Select the feature (column) for which you want to generate the histogram
feature = 'embed_0'

# Extract the values of the selected feature
feature_values = dataset[feature]

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(feature_values, bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of {feature}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate the mean, variance, skewness, and kurtosis of the feature
feature_mean = np.mean(feature_values)
feature_variance = np.var(feature_values)
feature_skewness = skew(feature_values)
feature_kurtosis = kurtosis(feature_values)

# Print descriptive statistics
print(f"Mean of {feature}: {feature_mean}")
print(f"Variance of {feature}: {feature_variance}")
print(f"Skewness of {feature}: {feature_skewness}")
print(f"Kurtosis of {feature}: {feature_kurtosis}")

# Plot box plot for the feature
plt.figure(figsize=(8, 6))
plt.boxplot(feature_values, vert=False)
plt.title(f'Box Plot of {feature}')
plt.xlabel('Value')
plt.grid(True)
plt.show()
