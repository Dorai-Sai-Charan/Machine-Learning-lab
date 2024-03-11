import numpy as np

def matrixmult(mat1, mat2):
    # Get the dimensions of the input matrices
    matdim1 = np.shape(mat1)
    row1, column1 = matdim1
    matdim2 = np.shape(mat2)
    row2, column2 = matdim2

    # Check if the matrices can be multiplied
    if column1 != row2:
        # If the number of columns in the first matrix is not equal to the number of rows in the second matrix,
        # multiplication is not possible, raise a TypeError
        raise TypeError("Matrices cannot be multiplied")
    else:
        # If matrices can be multiplied, perform the multiplication using np.dot and return the result
        return np.dot(mat1, mat2)

# Example matrices
mat1 = [[10, 11, 12], [12, 13, 14], [15, 16, 17]]
mat2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Print the result of matrix multiplication
print(matrixmult(mat1, mat2))
