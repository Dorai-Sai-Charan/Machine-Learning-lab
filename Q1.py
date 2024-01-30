def vowel_and_cons_count(in_str):
    vowel=0
    consonant=0
    in_str=in_str.lower()
    for alph in in_str:
        if alph in 'aeiou':
            vowel+=1
            
        elif alph.isalpha():#checks whether it is alpha or not
            consonant+=1

    return vowel,consonant

in_str=input("Please Enter the String")
vowels,consonants=vowel_and_cons_count(in_str)

print("The no of vowels in the given string are",vowels)
print("The no of consonants in the given string are",consonants)