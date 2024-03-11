def commondigits(in_list1, in_list2):
    # Convert input lists to sets to remove duplicate elements and enable set operations
    list1 = set(in_list1)
    list2 = set(in_list2)
    
    # Find the common elements between the sets
    common_digits = list1.intersection(list2)
    
    # Return the count of common elements
    return len(common_digits)

# Example lists
list1 = [5, 4, 3, 6, 7]
list2 = [1, 2, 3, 4, 5]

print("The first list is:", list1)
print("The second list is:", list2)

# Call the function and print the result
common_digit_count = commondigits(list1, list2)
print("The count of common elements:", common_digit_count)
