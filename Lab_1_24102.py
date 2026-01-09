"""
SUBJECT: 22AIE213 - Machine Learning 
TOPIC: Basics of Python Programming

NAME: Adina Sree Venkat Utham Kumar
Roll Number: BL.SC.U4AIE24102
Section: AIE - E
"""

"""
QUESTION 1
Write a program to count the number of vowels and consonants present in an input string.
"""

def counter(input):
    # Convert all characters to uppercase to prevent confusion
    A = str.upper(input)
    # Create a reference string for all vowels in all caps
    vowels = "AEIOU"
    # Make 2 counter variables for vowels and consonants respectively
    Vcount = 0 
    Ccount = 0
    # Trace through each character in the input and check if it is a vowel or a consonant else skip
    for i in A: 
        if i in vowels:
            Vcount += 1
        elif i.isalpha(): # isaplha() checks if the character is an alphabet or not
            Ccount += 1
        else:
            continue
    # return the counter variables
    return Vcount,Ccount

"""
QUESTION 2
Write a program that accepts two matrices A and B as input and returns their product AB. 
Check if A & B are multipliable; if not, return error message. 
"""

def matmult(A,B):
    # Get the number of rows and columns of both matrices to calculate order of matrices
    Arow = len(A)
    Acol = len(A[0])
    Brow = len(B)       
    Bcol = len(B[0])
    # Check for Order rule for matrix multiplication
    # The number of columns in the first matrix must be equal to the number of rows in the second matrix
    if Acol != Brow:
        return "Error: Matrices are not feasible for multiplication"
    else:
        # Create a null matrix with order Arow x Bcol to store the result of multiplication
        # We import numpy as np and make a null matrix using np.zeros(order)
        # Numpy is a module used for mathematical operations
        import numpy as np
        result = np.zeros((Arow,Bcol))
        # Loops to traverse through row column pairs of A and B and calculate the dot product
        for i in range(Arow):
            for j in range(Bcol):
                for k in range(Acol):
                    result[i][j] += A[i][k] * B[k][j]
        # We return the result matrix
        return result
    
"""
QUESTION 3
Write a program to find the number of common elements between two lists. The lists 
contain integers. 
"""

def common(a1,a2):
    # Convert both lists to sets to eliminate duplicate elements
    # Logic is that sets don't allow duplicate elements
    s1 = set(a1)
    s2 = set(a2)
    # Find the intersection of both sets to get common elements
    # We do that by using intersection() inbuilt method
    common = s1.intersection(s2)
    # Return the number of common elements using len() inbuilt function which gives number of elements in the list
    return len(common)

"""
QUESTION 4
Write a program that accepts a matrix as input and returns its transpose. 
"""

def transpose(A):
    # Get the number of rows and columns of the given matrix
    rows = len(A)
    cols = len(A[0])
    # Create a null matrix with order cols x rows to store the transpose of the matrix using above method
    import numpy as np
    result = np.zeros((cols,rows))
    # We use a loop to re allocate the values of original matrix to the new location of the transpose matrix
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    # Return the transpose matrix
    return result

"""
QUESTION 5
Generate a list of 100 random numbers between 100 and 150. Find the mean, median and mode for these numbers.
"""

def stats(vec):
    # Mean is sum of all elements divided by number of elements
    # Median is the middle element in a sorted list
    # Mode is the most frequently occurring element in the list
    # Statistics module is imported to make it easy to get needed values which has inbuilt functions for mean, median and mode
    import statistics as stats
    mean = stats.mean(vec)
    median = stats.median(vec)
    mode = stats.mode(vec)
    # Return the 3 values
    return mean, median, mode

"""
MAIN PROGRAM
"""

def main():
    
    # QUESTION 1
    vowels, consonants = counter(input("Enter a string: "))
    print("Number of Vowels:", vowels, "\nNumber of Consonants:", consonants)
    print()

    # QUESTION 2
    # Get order of matrices
    Arow = int(input("Enter number of rows for matrix A: "))
    Acol = int(input("Enter number of columns for matrix A: "))
    Brow = int(input("Enter number of rows for matrix B: "))
    Bcol = int(input("Enter number of columns for matrix B: "))
    import numpy as np
    print("Enter elements of Matrix A:")
    A = np.zeros((Arow,Acol))
    for i in range(Arow):
        for j in range(Acol):
            A[i][j] = int(input())
    print("Enter elements of Matrix B:")
    B = np.zeros((Brow,Bcol))
    for i in range(Brow):
        for j in range(Bcol):
            B[i][j] = int(input())
    result = matmult(A,B)
    print("Resultant Matrix after multiplication:")
    print(result)
    print()

    # QUESTION 3
    n1 = int(input("Enter number of elements in list 1: "))
    l1 = []
    for i in range(n1):
        l1.append(int(input()))
    n2 = int(input("Enter number of elements in list 2: "))
    l2 = []
    for i in range(n2):
        l2.append(int(input()))
    commonnums = common(l1,l2)
    print("Number of common elements:", commonnums)
    print()

    # QUESTION 4
    Mrow = int(input("Enter number of rows for the matrix: "))
    Mcol = int(input("Enter number of columns for the matrix: "))
    print("Enter elements of the Matrix:")
    M = np.zeros((Mrow,Mcol))
    for i in range(Mrow):
        for j in range(Mcol):
            M[i][j] = int(input())
    transposemat = transpose(M)
    print("Transpose of the Matrix:")
    print(transposemat)
    print()

    # QUESTION 5
    import random
    vec = []
    for i in range(100):
        vec.append(random.randint(100,150))
    mean, median, mode = stats(vec)
    print("List:", vec)
    print("Mean:", mean)
    print("Median:", median)
    print("Mode:", mode)

main()