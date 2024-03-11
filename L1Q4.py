import numpy as np

def mat_transpose(matr):
    # Return the transpose of the input matrix using np.transpose()
    return np.transpose(matr)

# Create a NumPy array
mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute the transpose of the matrix using the defined function
transmat = mat_transpose(mat)

# Print the original matrix and its transpose
print("The matrix is:", mat)
print("The transposed matrix is:", transmat)
