import numpy as np
import pandas as pd

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
    
    
    
