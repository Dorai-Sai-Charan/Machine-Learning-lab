def categorical_to_numerical(categories):
    # Dictionary to store the encoding of each category
    encoding_val = {}
    # Counter to keep track of the numerical encoding
    count = 0
    
    # Iterate through each category
    for i in categories:
        # Check if the category is not already encoded
        if i not in encoding_val:
            # Assign a numerical value to the category and increment the counter
            encoding_val[i] = count
            count += 1
    
    # Return a list of numerical values corresponding to the input categories
    return [encoding_val[i] for i in categories]

# Example categories
categories = ['One', 'Two', 'One', 'Three', 'Three', 'One', 'Two']

# Convert categorical values to numerical representations
encoded_values = categorical_to_numerical(categories)

# Print the encoded values
print("Encoded values:", encoded_values)
