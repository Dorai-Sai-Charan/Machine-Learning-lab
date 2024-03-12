import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
dataset = pd.read_excel("C:\\OS\\Machine learning\\Machine Learning Lab\\LAB4-20.02.24\\training_mathbert 2.xlsx")

# Extract columns starting with "embed_"
mathbert_columns = [col for col in dataset.columns if col.startswith("embed_")]
mathbert_data = dataset[mathbert_columns]

# Add the output column to mathbert_data
mathbert_data['output'] = dataset['output']

# Define valid output values
valid_output_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# Filter mathbert_data based on valid output values
mathbert_data = mathbert_data[mathbert_data['output'].isin(valid_output_values)]

# Group data by 'output'
grouped_data = mathbert_data.groupby('output')

# Calculate class centroids (mean vectors)
class_centroids = grouped_data.mean()

# Calculate class spreads (standard deviations)
class_spreads = grouped_data.std()

# Select two classes for interclass distance calculation (e.g., first two classes)
class_1 = class_centroids.iloc[0]
class_2 = class_centroids.iloc[1]

# Calculate the Euclidean distance between class centroids
interclass_distance = np.linalg.norm(class_1 - class_2)

# Output the results
print("Class Centroids:")
print(class_centroids)
print("\nClass Spreads:")
print(class_spreads)
print("\nInterclass Distance between Class 1 and Class 2:", interclass_distance)

# Plot class centroids for visualization
plt.figure(figsize=(8, 6))
plt.scatter(class_centroids.iloc[:, 0], class_centroids.iloc[:, 1], c=class_centroids.index, cmap='viridis', s=100)
plt.title('Class Centroids Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Output')
plt.grid(True)
plt.show()
