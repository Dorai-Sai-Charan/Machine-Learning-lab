import numpy as np

def mat_transpose(matr):
    return np.transpose(matr)

mat=np.array([[1,2,3],[4,5,6],[7,8,9]])

transmat=mat_transpose(mat)
print("The matrix is",mat)
print("The transposed matrix is",transmat)
