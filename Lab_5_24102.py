"""
SUBJECT: 22AIE213 - Machine Learning 
TOPIC: Classification, Regression, and K-Means Clustering with Scikit-Learn Models

NAME: Adina Sree Venkat Utham Kumar
Roll Number: BL.SC.U4AIE24102  
Section: AIE - E
"""

# Modules Used
# Pandas for data handling
import pandas as pd
# Numpy for numerical computations
import numpy as np
# Matplotlib for data visualization
import matplotlib.pyplot as plt
# Scikit-learn for models, metrics, and data splitting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Helper Functions
# Load Data from excel sheets
def load_data(path, sheet):
    # Load excel sheet
    return pd.read_excel(path, sheet_name=sheet)

# Split data into train and test sets
def split_train_test(features, target, test_size=0.2, random_state=42):
    # Split features and target into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# A1: Train Linear Regression Model
def train_linear_regression(X_train, y_train):
    # Train a linear regression model
    reg = LinearRegression().fit(X_train, y_train)
    return reg

# Predict using the model
def predict(model, X):
    # Make predictions using the trained model
    return model.predict(X)

# A2: Calculate Regression Metrics
def calculate_metrics(y_true, y_pred):
    # Calculate MSE, RMSE, MAPE, and R2 scores
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# A4: Perform K-Means Clustering
def perform_kmeans(X, n_clusters=2, random_state=0):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(X)
    return kmeans

# A5: Calculate Clustering Metrics
def calculate_clustering_metrics(X, labels):
    # Calculate Silhouette Score, CH Score, and DB Index
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    return sil_score, ch_score, db_index

# A6: Evaluate Clustering for Different K Values
def evaluate_kmeans_for_range(X, k_range):
    # Evaluate clustering metrics for a range of k values
    sil_scores = []
    ch_scores = []
    db_indices = []
    for k in k_range:
        kmeans = perform_kmeans(X, n_clusters=k, random_state=42)
        sil, ch, db = calculate_clustering_metrics(X, kmeans.labels_)
        sil_scores.append(sil)
        ch_scores.append(ch)
        db_indices.append(db)
    return sil_scores, ch_scores, db_indices

# Plot metrics against k
def plot_metrics_vs_k(k_range, sil_scores, ch_scores, db_indices):
    # Plot clustering metrics against k values
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('k values')
    ax1.set_ylabel('Silhouette Score / DB Index', color=color)
    ax1.plot(k_range, sil_scores, label='Silhouette Score', color='blue')
    ax1.plot(k_range, db_indices, label='DB Index', color='green')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('CH Score', color=color)
    ax2.plot(k_range, ch_scores, label='CH Score', color='red')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Clustering Metrics vs k')
    plt.show()

# A7: Elbow Plot for Optimal K
def elbow_plot(X, k_range):
    # Generate elbow plot using inertia
    distortions = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        distortions.append(kmeans.inertia_)
    plt.plot(k_range, distortions)
    plt.title('Elbow Plot')
    plt.xlabel('k values')
    plt.ylabel('Inertia')
    plt.show()
    return distortions

# MAIN FUNCTION
def main():
    file_path = "galaxy_features.xlsx"  
    sheet_name = "Galaxy_Features"  

    # Load data
    data = load_data(file_path, sheet_name)

    # Define features and target (for regression)
    # Adjust based on your dataset
    feature_cols = ["Feature1_Brightness", "Feature2_EdgeDensity"]  
    # Target for regression
    target_col = "Class_Label"  

    # Extract features and target
    # Use .values to get numpy array
    X = data[feature_cols].values  
    y = data[target_col].values

    # Split into train and test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # A1: Train with one attribute (e.g., first feature)
    # Reshape for sklearn which expects 2D array for features
    X_train_single = X_train[:, 0].reshape(-1, 1)  
    X_test_single = X_test[:, 0].reshape(-1, 1)
    model_single = train_linear_regression(X_train_single, y_train)
    y_train_pred_single = predict(model_single, X_train_single)
    y_test_pred_single = predict(model_single, X_test_single)

    # A2: Metrics for single attribute
    train_metrics_single = calculate_metrics(y_train, y_train_pred_single)
    test_metrics_single = calculate_metrics(y_test, y_test_pred_single)
    print("Single Attribute [Train Metrics] (MSE, RMSE, MAPE, R2):", train_metrics_single)
    print("Single Attribute [Test Metrics] (MSE, RMSE, MAPE, R2):", test_metrics_single)

    # A3: Train with all attributes
    model_all = train_linear_regression(X_train, y_train)
    y_train_pred_all = predict(model_all, X_train)
    y_test_pred_all = predict(model_all, X_test)

    # Metrics for all attributes
    train_metrics_all = calculate_metrics(y_train, y_train_pred_all)
    test_metrics_all = calculate_metrics(y_test, y_test_pred_all)
    print("All Attributes [Train Metrics] (MSE, RMSE, MAPE, R2):", train_metrics_all)
    print("All Attributes [Test Metrics] (MSE, RMSE, MAPE, R2):", test_metrics_all)

    # A4: K-means with k=2
    # For clustering: Remove target, use features only (X_train)
    kmeans_2 = perform_kmeans(X_train, n_clusters=2)
    print("K=2 Labels:", kmeans_2.labels_)
    print("K=2 Cluster Centers:", kmeans_2.cluster_centers_)

    # A5: Metrics for k=2
    metrics_2 = calculate_clustering_metrics(X_train, kmeans_2.labels_)
    print("K=2 Metrics (Silhouette, CH, DB):", metrics_2)

    # A6: Evaluate for k=2 to 10
    k_range = range(2, 11)
    sil_scores, ch_scores, db_indices = evaluate_kmeans_for_range(X_train, k_range)
    print("Silhouette Scores for k = [2-10]:", sil_scores)
    print("CH Scores for k = [2-10]:", ch_scores)
    print("DB Indices for k = [2-10]:", db_indices)
    plot_metrics_vs_k(k_range, sil_scores, ch_scores, db_indices)

    # A7: Elbow plot
    # As our current dataset is small, we can use the same range for k values
    elbow_plot(X_train, range(2, len(X_train)+1))

if __name__ == "__main__":
    main()