"""
SUBJECT: 22AIE213 - Machine Learning 
TOPIC: k-NN Classifier Implementation and Evaluation

NAME: Adina Sree Venkat Utham Kumar
Roll Number: BL.SC.U4AIE24102
Section: AIE - E
"""

# Modules Used
import numpy as np

# Dot Product Function
def dot(a,b):
    # zip function pairs elements from both lists and computes the sum of their products
    return sum(i*j for i,j in zip(a,b))

def euclidean(a,b):
    # Calculate the Euclidean distance between two points
    return np.sqrt(sum((i-j)**2 for i,j in zip(a,b)))

def main():
    # A1: DOT PRODUCT AND ECULIDEAN CALCULATION
    a = [1,2,3]
    b = [4,5,6]
    print("Dot Product of", a, "and", b, "is:", dot(a,b))
    print("Euclidean Distance between", a, "and", b, "is:", euclidean(a,b))

    # Compare with NumPy results
    print("NumPy Dot Product:", np.dot(a,b))
    print("NumPy Euclidean Distance:", np.linalg.norm(np.array(a)-np.array(b)))

    # A2: INTERCLASS AND INTRACLASS SPREAD CALCULATION

    # A3: HISTOGRAM PLOTTING

    # A4: MINOWSKI DISTANCE CALCULATION AND PLOTTING

    # A5: DISTANCES EVALUATION

    # A6: DIVIDE DATASET INTO TRAINING AND TEST SETS

    # A7: K-NN CLASSIFIER IMPLEMENTATION FOR TRAINING

    # A8: K-NN CLASSIFIER IMPLEMENTATION FOR TESTING

    # A9: k-NN PREDICTION ANALYSIS

    # A10: CREATION OF OWN k-NN CLASSIFIER FUNCTION

    # A11: ANALYSIS OF VARIOUS k VALUES

    # A12: CONFUSION MATRIX AND PERFORMANCE METRICS

    # A13: OWN CODE ANALYSIS

    # A14: COMAPARISON BETWEEN k-NN AND MATRIX INVERSE METHOD 

    # O1: ANALYSIS OF A NORMAL DISTRIBUTION

    # O2: ANALYSIS OF PERFORMANCE METRICS FOR DIFFERENT DATA SPLITS

    # O3: AUROC CALCULATION AND PLOTTING

main()