import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_excel("C:\\OS\\Machine learning\\Machine Learning Lab\\LAB4-20.02.24\\training_mathbert 2.xlsx")

# Select only the MathBERT embedding columns
mathbert_columns = [col for col in dataset.columns if col.startswith("embed_")]
mathbert_data = dataset[mathbert_columns]

# Add the 'output' column to the mathbert_data DataFrame
mathbert_data['output'] = dataset['output']

# Filter out rows where the 'output' column value is not within the desired range
valid_output_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
mathbert_data = mathbert_data[mathbert_data['output'].isin(valid_output_values)]

# Group the data by the 'output' column to create classes
grouped_data = mathbert_data.groupby('output')

# Calculate the mean for each class
class_centroids = grouped_data.mean()

# Calculate the spread (standard deviation) for each class
class_spreads = grouped_data.std()

# Select two classes for interclass distance calculation
class_1 = class_centroids.iloc[0]  # For example, choose the first class
class_2 = class_centroids.iloc[1]  # For example, choose the second class

# Calculate the distance between mean vectors between classes
interclass_distance = np.linalg.norm(class_1 - class_2)

# Output the results
print("Class Centroids:")
print(class_centroids)
print("\nClass Spreads:")
print(class_spreads)
print("\nInterclass Distance between Class 1 and Class 2:", interclass_distance)
