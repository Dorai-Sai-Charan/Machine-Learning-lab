import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = r"C:\OS\Machine learning\Machine Learning Lab\LAB5-27.02.24\Lab Session1 Data.xlsx"
df = pd.read_excel(file_path)

# Selecting relevant columns
columns_of_interest = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
X = df[columns_of_interest].values
y = df["Payment (Rs)"].values

# Dimensionality of the vector space
dimensionality = X.shape[1]

# Number of vectors
num_vectors = X.shape[0]

# Rank of Matrix A
rank_A = np.linalg.matrix_rank(X)

# Pseudo-Inverse to find the cost of each product
pseudo_inverse_X = np.linalg.pinv(X)
cost_of_products = np.dot(pseudo_inverse_X, y)

# Mark customers as RICH or POOR based on payment threshold
df['Customer_Type'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')

# Encode 'Customer_Type' to numerical values
label_encoder = LabelEncoder()
df['Customer_Type'] = label_encoder.fit_transform(df['Customer_Type'])

# Train-test split for classification
X_classification = df[columns_of_interest].values
y_classification = df['Customer_Type'].values
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Train a classifier model (e.g., Logistic Regression)
classifier_model = LogisticRegression()
classifier_model.fit(X_train, y_train)

# Predictions
y_pred = classifier_model.predict(X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print(f"Dimensionality of the vector space: {dimensionality}")
print(f"Number of vectors in the vector space: {num_vectors}")
print(f"Rank of Matrix X: {rank_A}")
print(f"Cost of each product (using Pseudo-Inverse):\n{cost_of_products}")
print("Classification Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
