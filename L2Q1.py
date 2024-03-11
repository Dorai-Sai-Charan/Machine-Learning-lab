def Euclidean_dist(vect1, vect2):
    # Check if the lengths of the vectors are equal
    if len(vect1) != len(vect2):
        print("Please provide vectors of equal lengths")
        
    # Initialize the Euclidean distance to 0
    eucdist = 0
    
    # Iterate over each element of the vectors
    for i in range(len(vect1)):
        # Add the square of the difference of corresponding elements to the Euclidean distance
        eucdist += (vect1[i] - vect2[i]) ** 2
        
    # Return the square root of the sum of squared differences
    return eucdist ** 0.5

def Manhattan_dist(vect1, vect2):
    # Check if the lengths of the vectors are equal
    if len(vect1) != len(vect2):
        print("Please provide vectors of equal lengths")
        
    # Initialize the Manhattan distance to 0
    mandist = 0
    
    # Iterate over each element of the vectors
    for i in range(len(vect1)):
        # Add the absolute difference of corresponding elements to the Manhattan distance
        mandist += abs(vect1[i] - vect2[i])
        
    # Return the sum of absolute differences
    return mandist

# Example vectors
vect1 = [1, 2, 3]
vect2 = [4, 5, 6]

# Calculate and print the Euclidean distance
print("The Euclidean distance of the given vectors is:", Euclidean_dist(vect1, vect2))

# Calculate and print the Manhattan distance
print("The Manhattan distance of the given vectors is:", Manhattan_dist(vect1, vect2))
