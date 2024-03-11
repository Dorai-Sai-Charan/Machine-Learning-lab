def onehot_encoding(categories):
    # Get the unique categories
    unique_categories = list(set(categories))
    # Dictionary to store the encoding for each category
    encoding = {}
    # Iterate over the unique categories
    for i, category in enumerate(unique_categories):
        # Create a binary vector where one element is 1 at the index corresponding to the category position
        encoding[category] = [0] * i + [1] + [0] * (len(unique_categories) - i - 1)
    # Return a list of one-hot encoded vectors for each category in the input
    return [encoding[category] for category in categories]

# Example categories
categories = ['One', 'Two', 'One', 'Three', 'Three', 'One', 'Two']

# Perform one-hot encoding
encoded_values = onehot_encoding(categories)

# Print the one-hot encoded values
print("One-hot encoded values are:")
for category, encoded in zip(categories, encoded_values):
    print(f"{category}: {encoded}")
