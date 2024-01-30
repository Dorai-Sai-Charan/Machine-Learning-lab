import numpy as np

def matrixmult(mat1,mat2):
    matdim1=np.shape(mat1)
    row1,column1=matdim1
    matdim2=np.shape(mat2)
    row2,column2=matdim2
    if column1 != row2:
        raise TypeError("matrices cannot be multiplied")
    else:
        return np.dot(mat1,mat2)
    
mat1=[[10,11,12],[12,13,14],[15,16,17]]
mat2=[[1,2,3],[4,5,6],[7,8,9]]
print(matrixmult(mat1,mat2))