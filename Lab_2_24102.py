"""
SUBJECT: 22AIE213 - Machine Learning 
TOPIC: Data Interpolation and Matrix Operations, Introduction to NumPy Package

NAME: Adina Sree Venkat Utham Kumar
Roll Number: BL.SC.U4AIE24102
Section: AIE - E
"""

# Modules Used
# Pandas for data handling
import pandas as pd
# Numpy for numerical computations
import numpy as np
# Seaborn and Matplotlib for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Helper Functions
# Load Data from excel sheets
def load_data(path, sheet):
    """Load excel sheet."""
    return pd.read_excel(path, sheet_name=sheet)

# Display quick info about the table
def show_info(table):
    print("\nColumns:", list(table.columns))
    print("Missing values:\n", table.isna().sum())
    print("Stats:\n", table.describe())

# A1: FEATURE MATRIX AND COST VECTOR CALCULATION

# Split features and target [i.e. Result Column]
def split_features_target(table, feature_cols, target_col):
    # to_numpy() method is used to convert DataFrame to NumPy array
    features = table[feature_cols].to_numpy()
    target = table[target_col].to_numpy()
    return features, target

# Rank calculation
def get_rank(features):
    # Linalg package is used for linear algebra operations
    return np.linalg.matrix_rank(features)

def get_cost(features, target):
    # Pseudo-Inverse calculation
    # '@' operator is used for matrix multiplication
    # This is X = (A^T * A)^-1 * A^T * B
    return np.linalg.pinv(features) @ target

# A2: CUSTOMER CLASSIFICATION

# Mark rich or poor based on limit, here limit is 200
def mark_rich_or_poor(table, target_col, limit=200):
    # We use list comprehension to create a new column based on the condition
    table["Class"] = ["RICH" if p > limit else "POOR" for p in table[target_col]]
    return table

# A3: IRCTC STATS AND VISUALIZATION

# Mean calculation
def mean(arr):
    return sum(arr) / len(arr)

# Variance calculation using Sum(x - mean)^2 / N
def var(arr):
    m = mean(arr)
    return sum((x-m)**2 for x in arr) / len(arr)

# Selection of particular data points using List Filtering
def pick_wednesday(table):
    # pd.to_datetime converts string dates to datetime objects
    # dt.day_name() gives the weekday name
    table["weekday"] = pd.to_datetime(table["Date"]).dt.day_name()
    return table[table["weekday"] == "Wednesday"]

def pick_month(table, month=4):
    table["month"] = pd.to_datetime(table["Date"]).dt.month
    return table[table["month"] == month]

# Probability of negative change (loss)
def prob_negative(arr):
    return np.sum(arr < 0) / len(arr)

# Selecttion of particular data points using List Filtering
def add_weekday(t):
    t["weekday"] = pd.to_datetime(t["Date"]).dt.day_name()
    return t

# Probability of profit on Wednesday
def prob_wed_profit(t):
    # P(profit on Wednesday).
    wed = t[t["weekday"] == "Wednesday"].copy()
    if len(wed) == 0:
        return 0
    return sum(wed["Chg%"] > 0) / len(wed)

# Conditional Probability of profit given Wednesday
def cond_wed_profit(t):
    # P(profit | Wednesday).
    return prob_wed_profit(t)

# Scatter plot of Chg% vs weekday
def plot_wday_vs_chg(t):
    plt.scatter(t["weekday"], t["Chg%"])
    plt.title("Chg% vs Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Chg%")
    plt.show()

# A5–A7: SIMILARITY MEASURES
# a and b are binary numpy arrays

# Jaccard Similarity
def jaccard(a, b):
    f11 = np.sum((a==1)&(b==1))
    f10 = np.sum((a==1)&(b==0))
    f01 = np.sum((a==0)&(b==1))
    denom = f11 + f10 + f01
    return f11/denom if denom != 0 else 0

# Simple Matching Coefficient
def smc(a, b):
    total = len(a)
    matches = np.sum(a==b)
    return matches/total if total != 0 else 0

# Cosine Similarity
def cosine(a, b):
    # We normalize the vectors and compute dot product
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0   # safe fallback to prevent division by zero [NaN]
    return np.dot(a, b) / (na * nb)

# Analyze similarity between first two rows
def compare_similarity(data):
    """Use first 2 rows."""
    a = data[0]
    b = data[1]
    print("Jaccard:", jaccard(a,b))
    print("SMC:", smc(a,b))
    print("Cosine:", cosine(a,b))

# Heatmap visualization using seaborn
def make_heatmap(data, title):
    sns.heatmap(data, annot=False)
    plt.title(title)
    plt.show()

# Pairwise similarity matrix
# Basically, we compute similarity between each pair of rows
# This function speeds up the process by limiting to first 20 rows
def make_pairwise(data, measure="cosine"):
    # We apply on first 20 rows for simplicity
    data = data[:20]   
    n = len(data)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if measure=="cosine":
                mat[i,j] = cosine(data[i], data[j])
            elif measure=="jaccard":
                mat[i,j] = jaccard(data[i], data[j])
            else:
                mat[i,j] = smc(data[i], data[j])
    return mat

# A8: SIMPLE IMPUTATION
# An imputation is filling missing values with some statistics based on other values
# Data types are of two types: Categorical (text) and Numerical (numbers)

# Fill missing values with mode for categorical and median for numerical
def fill_missing(table):
    for col in table.columns:
        # Here object dtype implies categorical data
        # fillna() method fills missing values as 0 for numerical and mode for categorical
        if table[col].dtype == 'object':
            table[col] = table[col].fillna(table[col].mode()[0])
        else:
            table[col] = table[col].fillna(table[col].median())
    return table

# A9: SIMPLE NORMALIZATION
def normalize(table, cols):
    # Apply min-max normalization
    # Formula: (x - min) / (max - min)
    for col in cols:
        table[col]=(table[col]-table[col].min())/(table[col].max()-table[col].min())
    return table

# O1–O3 OPTIONAL

# Here we create two square matrices from the table
# Then we compute pairwise similarity on random and marketing data
def make_square_sets(table, size=5):
    a = table.iloc[:size,:size].to_numpy()
    b = table.iloc[-size:,-size:].to_numpy()
    print("O1 squares made.")
    return a,b

# Random Similarity implies that we take random samples from the data and compute similarity
def random_similarity(table, size=20):
    numeric = table.select_dtypes(include=[np.number])
    
    # Convert to binary using median
    binary = (numeric > numeric.median()).astype(int)
    
    # Remove columns that are constant (all 0 or all 1)
    binary = binary.loc[:, binary.nunique() > 1]
    
    # Sample after cleaning
    sample = binary.sample(size)
    
    # Heatmap
    make_heatmap(make_pairwise(sample.values,"cosine"), "Random Cosine (O2)")
    make_heatmap(make_pairwise(sample.values,"jaccard"), "Random Jaccard (O2)")

# Marketing Similarity implies that we binarize the data based on median and compute similarity
# Why do we do this is because marketing data often has outliers
# That is some values are extremely high or low compared to the rest
# This happens in cases like customer spending, where a few customers spend a lot more than others
# Binarization: values above median -> 1, below median -> 0
# Median is used to reduce the effect of outliers
# Outliers are extreme values that can skew the data
# Skewing data means that the data distribution is not uniform

def marketing_similarity(table):
    numeric = table.select_dtypes(include=[np.number])
    binary = (numeric > numeric.median()).astype(int)
    compare_similarity(binary.values)
    make_heatmap(make_pairwise(binary.values,"cosine"), "Marketing Cosine (O3)")

# MAIN FUNCTION
def main():
    file_path = "Lab Session Data.xlsx"

    # A1: FEATURE MATRIX AND COST VECTOR CALCULATION
    purchase = load_data(file_path,"Purchase data")
    feats = ["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]
    target = "Payment (Rs)"
    fmat, payments = split_features_target(purchase, feats, target)
    print("Matrix Rank:", get_rank(fmat))
    print("Product Cost:", get_cost(fmat, payments))
    
    # A2: CUSTOMER CLASSIFICATION
    purchase = mark_rich_or_poor(purchase, target)
    print(purchase[["Payment (Rs)","Class"]])

    # A3: STOCK PRICE ANALYSIS
    irctc = load_data(file_path, "IRCTC Stock Price")
    irctc = add_weekday(irctc)

    prices = irctc["Price"].values
    print("Mean:", mean(prices), "Var:", var(prices))
    print("Prob loss overall:", prob_negative(irctc["Chg%"].values))

    print("P(profit on Wed):", prob_wed_profit(irctc))
    print("P(profit | Wed):", cond_wed_profit(irctc))

    plot_wday_vs_chg(irctc)
    
    # A4: DATA LOADING AND INFORMATION
    thyroid = load_data(file_path,"thyroid0387_UCI")
    show_info(thyroid)
    
    # A5–A7: SIMILARITY MEASURES
    nums = thyroid.select_dtypes(include=[np.number])
    binary = (nums > nums.median()).astype(int)
    compare_similarity(binary.values)
    make_heatmap(make_pairwise(binary.values,"cosine"),"Thyroid Cosine")

    # A8: SIMPLE IMPUTATION
    thyroid = fill_missing(thyroid)
    
    # A9: SIMPLE NORMALIZATION
    numeric_cols = thyroid.select_dtypes(include=[np.number]).columns
    thyroid = normalize(thyroid, numeric_cols)

    # O1: SQUARE SET CREATION
    make_square_sets(purchase)

    # O2: RANDOM SIMILARITY
    random_similarity(thyroid)

    # O3: MARKETING SIMILARITY
    marketing = load_data(file_path,"marketing_campaign")
    marketing_similarity(marketing)

if __name__=="__main__":
    main()
