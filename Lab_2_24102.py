"""
SUBJECT: 22AIE213 - Machine Learning 
TOPIC: Data Interpolation and Matrix Operations, Introduction to NumPy Package

NAME: Adina Sree Venkat Utham Kumar
Roll Number: BL.SC.U4AIE24102
Section: AIE - E
"""

"""
QUESTION A1
Make 2 matrices of order 10x3 and 3x1 which are the observation vector and the output vector respectively.

Thinking Questions:
What is the dimensionality of the vector space for this data?
It is 3-dimensional as there are 3 features in the observation vector.

How many vectors exist in this vector space? 
There are 10 vectors in this vector space as there are 10 observations.

What is the rank of Matrix A? 
Rank of Matrix A is 3 as there are 3 linearly independent columns in the matrix.

How to calculate rank of a matrix? Calculate the rank of the feature matrix.
Rank of a matrix can be calculated using linear algebra concepts. 
It is the maximum number of linearly independent row or column vectors in the matrix. 
Rank of the matrix is 3 as there are 3 columns which are linearly independent.

Coding Questions:
Read the data from the csv file and create the observation and output matrices.
Calculate the rank of the feature matrix with numpy package. 
Using Pseudo-Inverse find the cost of each product available for sale.  
Display the results
"""
import numpy as np
import pandas as pd

# Read data from CSV file [only the relevant columns]
data = pd.read_excel("Lab Session Data.xlsx", sheet_name='Purchase data',usecols='A:E')

# Create observation matrix A (10x3) and output matrix B (10x1)
A = data.iloc[:,1:4].values  # First three columns as observation matrix
B = data.iloc[:, 4].values.reshape(-1, 1)  # Last column as output matrix

# Display the matrices
print("\nObservation Matrix A (10x3):\n", A)
print("\nOutput Matrix B (10x1):\n", B)

# Calculate the rank of the feature matrix A
rank = np.linalg.matrix_rank(A)
print("\nRank of Matrix A:", rank)

# Calculate the pseudo-inverse of matrix A
pseudo = np.linalg.pinv(A.flatten().reshape(10,3))

# Calculate the cost of each product (weights)
weights = np.dot(pseudo, B)
print("\nCost of each product:\n", (weights.flatten()))  # Flatten for better display

# Check for correctness by predicting output
predict = np.dot(A, weights)
print("\nPredicted Output:\n", predict.flatten())  # Flatten for better display

"""
QUESTION A2
Mark all customers (in “Purchase Data” table) with payments above Rs. 200 as RICH and others as POOR.
Develop a classifier model to categorize customers into RICH or POOR class based on purchase behavior. 
"""
# We create a new column 'Class' based on the payment condition
# Select the payment column and applying condition
data['Class'] = data.iloc[:,4].apply(lambda x: 'RICH' if x > 200 else 'POOR') 

# Display relevant columns with classification
print("\nCustomer Classification based on Payment:\n",data[['Customer','Class']])  

"""
QUESTION A3
"""
