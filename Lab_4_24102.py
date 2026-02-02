"""
SUBJECT: 22AIE213 - Machine Learning 
TOPIC: Regression Models

NAME: Adina Sree Venkat Utham Kumar
Roll Number: BL.SC.U4AIE24102
Section: AIE - E
"""

# k-Nearest Neighbors (kNN) is a supervised learning algorithm used for classification and regression.
# It classifies a new data point based on the majority class of its k closest neighbors in the feature space,
# using distance metrics like Euclidean or Minkowski.

# Imported Modules
# numpy: For numerical computations and array operations.
import numpy as np
# pandas: For data manipulation and reading Excel files.
import pandas as pd
# matplotlib.pyplot: For plotting graphs and histograms.
import matplotlib.pyplot as plt
# sklearn.neighbors: For kNN classifier.
from sklearn.neighbors import KNeighborsClassifier
# sklearn.model_selection: For train-test split and GridSearchCV.
from sklearn.model_selection import train_test_split, GridSearchCV
# sklearn.metrics: For confusion matrix, precision, recall, f1.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# A1: CONFUSION MATRIX AND METRICS WITH DATA FITTING

# A2: MSE, RMSE, MAPE, R2 SCORE FOR REGRESSION MODELS

# A3: SCATTER PLOT OF A RANDOM TRAINING DATA

# A4: SCATTER PLOT OF A RANDOM TESTING DATA USING KNN CLASSIFIER

# A5: DIVERSITY OF K VALUES FOR A4

# A6: TESTING OF KNN ON PROJECT DATA FOR A3 TO A5

# A7: HYPERPARAMETER TUNING USING GRIDSEARCHCV

def main():


main()