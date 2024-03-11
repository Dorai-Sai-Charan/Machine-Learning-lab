def vowel_and_cons_count(in_str):
    # Initialize counts for vowels and consonants
    vowel = 0
    consonant = 0
    
    # Convert the input string to lowercase to make the comparison case-insensitive
    in_str = in_str.lower()
    
    # Iterate through each character in the string
    for char in in_str:
        # Check if the character is a vowel
        if char in 'aeiou':
            vowel += 1
        # If the character is not a vowel and is alphabetic, count it as a consonant
        elif char.isalpha():  # checks whether it is alphabet or not
            consonant += 1
    
    # Return the counts of vowels and consonants
    return vowel, consonant

def main():
    # Prompt the user to enter a string and remove leading/trailing whitespaces
    in_str = input("Please Enter the String: ").strip()
    
    # Check if the input string is not empty
    if in_str:
        # Call the vowel_and_cons_count function to get counts of vowels and consonants
        vowels, consonants = vowel_and_cons_count(in_str)
        
        # Print the counts of vowels and consonants
        print("The number of vowels in the given string are:", vowels)
        print("The number of consonants in the given string are:", consonants)
    else:
        # If the input string is empty, prompt the user to enter a valid string
        print("Please enter a valid string.")

if __name__ == "__main__":
    # Call the main function when the script is executed
    main()
