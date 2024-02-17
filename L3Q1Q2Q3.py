import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset=pd.read_excel('C:\OS\Machine Learning Lab\LAB3-13.02.24\Lab Session1 Data.xlsx')

a=dataset.head()
print(a)


A = dataset[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values

C = dataset[['Payment (Rs)']].values

print(A)
print(C)

rows, cols = A.shape



print("The Dimensionality of the vector space:", cols)
print("Number of vectors are:", rows)

matrix = dataset.to_numpy()

rank = np.linalg.matrix_rank(A)
print("The rank of matrix A:", rank)

pinv_A = np.linalg.pinv(A)
X=pinv_A@C
print("The individual cost of a candy is: ",round(X[0][0]))
print("The individual cost of a mango is: ",round(X[1][0]))
print("The individual cost of a milk packet is: ",round(X[2][0]))
    
    
    
def classifier(df):
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    df['Predicted Category'] = classifier.predict(X)
    return df
# Load Excel file into a pandas DataFrame
df = pd.read_excel('Lab Session1 Data.xlsx')

df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
df=classifier(df)
print(df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']])
