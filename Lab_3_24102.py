"""
SUBJECT: 22AIE213 - Machine Learning 
TOPIC: KNN CLASSIFICATION MODEL

NAME: Adina Sree Venkat Utham Kumar
ROLL NUMBER: BL.SC.U4AIE24102
SECTION: AIE - E
"""
"""
k-Nearest Neighbors (kNN) is a supervised learning algorithm used for classification and regression.
It classifies a new data point based on the majority class of its k closest neighbors in the feature space
using distance metrics like Euclidean or Minkowski.
""" 

# IMPORTED MODULES
# numpy: For numerical computations and array operations
import numpy as np
# pandas: For data manipulation and reading Excel files
import pandas as pd
# matplotlib.pyplot: For plotting graphs and histograms
import matplotlib.pyplot as plt
# sklearn.neighbors: For kNN classifier
from sklearn.neighbors import KNeighborsClassifier
# sklearn.model_selection: For train-test split
from sklearn.model_selection import train_test_split
# sklearn.metrics: For accuracy, confusion matrix, precision, recall, f1, roc_auc, roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, fbeta_score, roc_auc_score, roc_curve
# scipy.spatial.distance: For Minkowski distance
from scipy.spatial.distance import minkowski
# scipy.stats: For normal distribution in optional question
from scipy.stats import norm

# Data Preprocessing
def get_data(file_path, sheet_name):
    # Read Excel file into DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Drop rows with missing values
    df = df.dropna()
    # Parse Volume: convert K to *1000, M to *1000000
    def parse_vol(v):
        if isinstance(v, str):
            if 'K' in v:
                return float(v.replace('K', '')) * 1000
            elif 'M' in v:
                return float(v.replace('M', '')) * 1000000
        return float(v)
    df['Volume'] = df['Volume'].apply(parse_vol)
    # Features: Low, High, Volume
    features = df[['Low', 'High', 'Volume']].values
    # Labels: 1 (green/safe) if Chg% > 0 else 0 (red/risky)
    labels = np.where(df['Chg%'] > 0, 1, 0)
    return features, labels

# A1: DOT PRODUCT AND LENGTH CALCULATION
def dot(a, b):
    # Manual dot product: sum of element-wise multiplication
    return sum(ai * bi for ai, bi in zip(a, b))

def euclid_norm(v):
    # Manual Euclidean norm: sqrt of sum of squares
    return sum(vi ** 2 for vi in v) ** 0.5

def comp_vec(a, b):
    # Compute manual and NumPy dot product
    man_dot = dot(a, b)
    np_dot = np.dot(a, b)
    # Compute manual and NumPy length
    man_len_a = euclid_norm(a)
    np_len_a = np.linalg.norm(a)
    man_len_b = euclid_norm(b)
    np_len_b = np.linalg.norm(b)
    return man_dot, np_dot, man_len_a, np_len_a, man_len_b, np_len_b

# A2: STATISTICAL DATA
def mean(data):
    # Manual mean: sum divided by count
    return sum(data) / len(data) if len(data) > 0 else 0

def var(data, mean_val):
    # Manual variance: average of squared differences from mean
    return sum((x - mean_val) ** 2 for x in data) / len(data) if len(data) > 0 else 0

def std(data, mean_val):
    # Standard deviation: square root of variance
    return var(data, mean_val) ** 0.5

def class_stat(features, labels, class_val):
    # Filter features for given class
    class_data = features[labels == class_val]
    # Mean per feature (column-wise)
    means = [mean(col) for col in class_data.T]
    # Std per feature
    stds = [std(col, means[i]) for i, col in enumerate(class_data.T)]
    return np.array(means), np.array(stds)

def class_dist(mean1, mean2):
    # Euclidean distance between two means
    return euclid_norm(mean1 - mean2)

# A3: HISTOGRAM FOR MEAN AND VARIANCE
def hist_plot(feature):
    # Compute histogram data
    hist, bins = np.histogram(feature, bins=10)
    # Plot histogram
    plt.hist(feature, bins=10)
    plt.title('Feature Histogram')
    plt.show()
    # Mean and variance of feature
    mean_val = mean(feature)
    var_val = var(feature, mean_val)
    return hist, bins, mean_val, var_val

# A4: MINKOWSKI DISTANCE PLOTTING
def mink_dist(a, b, p):
    # Manual Minkowski: sum of |ai - bi|^p raised to 1/p
    return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)

def mink_plot(a, b):
    # Distances for p from 1 to 10
    dists = [mink_dist(a, b, p) for p in range(1, 11)]
    # Plot distances vs p.
    plt.plot(range(1, 11), dists)
    plt.title('Minkowski Distance vs p')
    plt.show()
    return dists

# A5: COMPARISION OF MIKOWSKI DISTANCES
def comp_mink(a, b, p):
    # Manual and SciPy Minkowski
    man = mink_dist(a, b, p)
    sci_dist = minkowski(a, b, p)
    return man, sci_dist

# A6: DATA SPLITTING
def data_split(features, labels):
    # Split 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    # Further split train: 60% sub-train, 40% validation
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
    return X_subtrain, X_val, X_test, y_subtrain, y_val, y_test

# A7: KNN TRAINING
def knn_train(X_train, y_train, k=3):
    # Create and fit kNN classifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    return neigh

# A8: ACCURACY TESTING
def acc_test(model, X_test, y_test):
    # Get accuracy score
    return model.score(X_test, y_test)

# A9: CLASS PREDICTION
def class_pred(model, test_vec):
    # Predict class for test vector
    return model.predict([test_vec])[0]

# A10: MANUAL CODING OF KNN FUNCTION
def knn_man(X_train, y_train, test_vec, k=3):
    # Compute distances to all train points
    # Here, Euclidean distance
    dists = [mink_dist(test_vec, x, 2) for x in X_train]  
    # Get indices of k smallest distances
    idx = np.argsort(dists)[:k]
    # Majority class among k neighbors
    classes = y_train[idx]
    return np.bincount(classes).argmax()

def comp_clf(X_train, y_train, X_test, y_test, k=3):
    # Train package kNN
    model = knn_train(X_train, y_train, k)
    pkg_pred = model.predict(X_test)
    pkg_acc = accuracy_score(y_test, pkg_pred)
    # Manual kNN predictions
    man_pred = [knn_man(X_train, y_train, x, k) for x in X_test]
    man_acc = sum(man_pred == y_test) / len(y_test)
    return pkg_acc, man_acc

# A11: VARIATION OF K VALUES AND ACCURACY PLOTTING
def k_vary(X_train, y_train, X_test, y_test):
    # Accuracies for k=1 to 11
    accs = []
    for k in range(1, 12):
        model = knn_train(X_train, y_train, k)
        accs.append(acc_test(model, X_test, y_test))
    # Plot accuracies vs k
    plt.plot(range(1, 12), accs)
    plt.title('Accuracy vs k')
    plt.show()
    return accs

# A12 and A13: CONFUSION MATRIX AND PERFORMANCE METRICS
def conf_get(y_true, y_pred):
    # Manual confusion matrix (2x2 for binary)
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def acc_calc(conf):
    # Accuracy: (TP + TN) / total
    return (conf[1][1] + conf[0][0]) / np.sum(conf)

def prec_calc(conf):
    # Precision: TP / (TP + FP)
    return conf[1][1] / (conf[1][1] + conf[0][1]) if (conf[1][1] + conf[0][1]) > 0 else 0

def rec_calc(conf):
    # Recall: TP / (TP + FN)
    return conf[1][1] / (conf[1][1] + conf[1][0]) if (conf[1][1] + conf[1][0]) > 0 else 0

def fbeta_calc(conf, beta=1):
    # F-beta: (1 + beta^2) * (prec * rec) / (beta^2 * prec + rec)
    prec = prec_calc(conf)
    rec = rec_calc(conf)
    if prec + rec == 0:
        return 0
    return (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)

def met_eval(model, X, y, is_train=False):
    # Predict and get confusion
    pred = model.predict(X)
    conf = conf_get(y, pred)
    acc = acc_calc(conf)
    prec = prec_calc(conf)
    rec = rec_calc(conf)
    f1 = fbeta_calc(conf, beta=1)
    # Infer fit: compare train/test, but here returns metrics
    return conf.tolist(), float(acc), float(prec), float(rec), float(f1)

# A14: COMAPRISION WITH MATRIX INVERSION
def inv_comp():
    # Placeholder: Matrix inversion not implemented for classification
    return "kNN vs Inversion: kNN is instance-based, inversion for linear models."

# O1: NORMAL DISTRIBUTION PLOT
def norm_dist(data):
    # Generate normal data with same mean/std
    mean_val = np.mean(data)
    std_val = np.std(data)
    norm_data = np.random.normal(mean_val, std_val, len(data))
    # Plot histogram of data
    plt.hist(data, bins=10, density=True, alpha=0.6, color='g')
    # Plot normal PDF
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, norm.pdf(x, mean_val, std_val))
    plt.title('Histogram vs Normal')
    plt.show()
    return norm_data

# O2: COMPARE VARIATIONS IN DISTANCE METRICS
def met_vary(X_train, y_train, X_test, y_test):
    # Metrics: euclidean, manhattan, chebyshev
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    accs = []
    for m in metrics:
        model = KNeighborsClassifier(n_neighbors=3, metric=m)
        model.fit(X_train, y_train)
        accs.append(model.score(X_test, y_test))
    return accs

# O3: AUROC PLOTTING
def auroc_plot(model, X_test, y_test):
    # Probabilities for class 1
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    # Plot ROC
    plt.plot(fpr, tpr)
    plt.title(f'AUROC: {auc}')
    plt.show()
    return auc

def main():
    file_path = 'Lab Session Data.xlsx'
    sheet_name = 'IRCTC Stock Price'
    features, labels = get_data(file_path, sheet_name)
    
    # A1: DOT PRODUCT AND LENGTH CALCULATION
    a, b = features[0], features[1]
    dot_res = comp_vec(a, b)
    print("A1 Results:", dot_res)
    
    # A2: STATISTICAL DATA
    mean0, std0 = class_stat(features, labels, 0)
    mean1, std1 = class_stat(features, labels, 1)
    dist = class_dist(mean0, mean1)
    print("A2 Means:", mean0.tolist(), mean1.tolist())
    print("A2 Stds:", std0.tolist(), std1.tolist())
    print("A2 Distance:", dist)
    
    # A3: HISTOGRAM FOR MEAN AND VARIANCE
    hist_res = hist_plot(features[:, 0])
    print("A3 Mean/Var:", hist_res[2], hist_res[3])
    
    # A4: MINKOWSKI DISTANCE PLOTTING
    mink_res = mink_plot(a, b)
    print("A4 Dists:", mink_res)
    
    # A5: COMPARISION OF MIKOWSKI DISTANCES
    comp_res = comp_mink(a, b, 3)
    print("A5 Manual/SciPy:", comp_res)
    
    # A6: DATA SPLITTING
    X_subtrain, X_val, X_test, y_subtrain, y_val, y_test = data_split(features, labels)
    
    # A7: KNN TRAINING WITH K=3
    model = knn_train(X_subtrain, y_subtrain, 3)
    
    # A8: ACCURACY TESTING
    acc = acc_test(model, X_test, y_test)
    print("A8 Accuracy:", acc)
    
    # A9: CLASS PREDICTION
    pred = class_pred(model, X_test[0])
    print("A9 Prediction:", pred)
    
    # A10: MANUAL CODING OF KNN FUNCTION
    comp_res = comp_clf(X_subtrain, y_subtrain, X_test, y_test, 3)
    print("A10 Pkg/Manual Acc:", comp_res)
    
    # A11: VARIATION OF K VALUES AND ACCURACY PLOTTING
    accs_k = k_vary(X_subtrain, y_subtrain, X_test, y_test)
    print("A11 Accs:", accs_k)
    
    # A12 and A13: CONFUSION MATRIX AND PERFORMANCE METRICS
    train_met = met_eval(model, X_subtrain, y_subtrain, True)
    test_met = met_eval(model, X_test, y_test)
    print("A12 Train Metrics:", train_met)
    print("A12 Test Metrics:", test_met)
    
    # A14: COMAPRISION WITH MATRIX INVERSION
    comp_res = inv_comp()
    print("A14:", comp_res)
    
    # O1: NORMAL DISTRIBUTION PLOT
    norm_d = norm_dist(features[:, 0])
    
    # O2: COMPARE VARIATIONS IN DISTANCE METRICS
    met_accs = met_vary(X_subtrain, y_subtrain, X_test, y_test)
    print("O2 Accs:", met_accs)
    
    # O3: AUROC PLOTTING
    auc = auroc_plot(model, X_test, y_test)
    print("O3 AUC:", auc)

main()
