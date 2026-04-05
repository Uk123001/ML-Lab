"""
Galaxy10 DECaLS - Classification Pipeline
==========================================
A2: RandomizedSearchCV hyperparameter tuning
A3: SVM, Decision Tree, Random Forest, CatBoost, AdaBoost, XGBoost, Naive Bayes, MLP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import shap
import lime
import lime.lime_tabular

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading features...")
df = pd.read_csv("galaxy10_features.csv")

# Clean NaN and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop(columns=["label", "class_name"]).values
y = df["label"].values
feature_names = df.drop(columns=["label", "class_name"]).columns.tolist()

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "Inbetween Round Smooth",
    "Cigar Shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "EdgeOn No Bulge", "EdgeOn With Bulge"
]

print(f"Dataset shape: {X.shape}")

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    results = {
        "Model"          : name,
        "Train Accuracy" : round(accuracy_score(y_train, train_pred), 4),
        "Test Accuracy"  : round(accuracy_score(y_test,  test_pred),  4),
        "Test Precision" : round(precision_score(y_test, test_pred, average="weighted", zero_division=0), 4),
        "Test Recall"    : round(recall_score(y_test,    test_pred, average="weighted", zero_division=0), 4),
        "Test F1"        : round(f1_score(y_test,        test_pred, average="weighted", zero_division=0), 4),
    }
    return results, test_pred


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# 4. A2 + A3: MODELS AND HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("A2: Hyperparameter Tuning with RandomizedSearchCV")
print("A3: Training and Evaluating All Classifiers")
print("="*60)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models_and_grids = [
    (
        "SVM",
        SVC(random_state=42, probability=True),
        {
            "C"      : [0.1, 1, 10, 100],
            "kernel" : ["rbf", "linear", "poly"],
            "gamma"  : ["scale", "auto"],
        }
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth"        : [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "criterion"        : ["gini", "entropy"],
        }
    ),
    (
        "Random Forest",
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators"     : [100, 200, 300],
            "max_depth"        : [10, 20, None],
            "min_samples_split": [2, 5],
            "max_features"     : ["sqrt", "log2"],
        }
    ),
    (
        "CatBoost",
        CatBoostClassifier(random_state=42, verbose=0),
        {
            "iterations"   : [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth"        : [4, 6, 8],
        }
    ),
    (
        "AdaBoost",
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators" : [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        }
    ),
    (
        "XGBoost",
        XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
        {
            "n_estimators" : [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth"    : [3, 5, 7],
            "subsample"    : [0.7, 0.9, 1.0],
        }
    ),
    (
        "Naive Bayes",
        GaussianNB(),
        {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        }
    ),
    (
        "MLP",
        MLPClassifier(random_state=42, max_iter=300),
        {
            "hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
            "activation"        : ["relu", "tanh"],
            "learning_rate_init": [0.001, 0.01],
        }
    ),
]

# ─────────────────────────────────────────────
# 5. TRAIN, TUNE AND EVALUATE
# ─────────────────────────────────────────────
all_results = []
best_models = {}

for model_name, model, param_grid in models_and_grids:
    print(f"\nTuning {model_name}...")

    search = RandomizedSearchCV(
        estimator           = model,
        param_distributions = param_grid,
        n_iter              = 10,
        cv                  = cv,
        scoring             = "accuracy",
        n_jobs              = -1,
        random_state        = 42,
        verbose             = 0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_models[model_name] = best_model

    print(f"  Best params : {search.best_params_}")
    print(f"  CV accuracy : {search.best_score_:.4f}")

    results, test_pred = evaluate_model(
        model_name, best_model, X_train, X_test, y_train, y_test
    )
    all_results.append(results)

    print(f"  Train Accuracy : {results['Train Accuracy']}")
    print(f"  Test Accuracy  : {results['Test Accuracy']}")
    print(f"  Test F1        : {results['Test F1']}")

    plot_confusion_matrix(y_test, test_pred, model_name)

# ─────────────────────────────────────────────
# 6. RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY TABLE")
print("="*60)

results_df = pd.DataFrame(all_results).set_index("Model")
print(results_df.to_string())
results_df.to_csv("classification_results.csv")
print("\nSaved to classification_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics_test = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
results_df[metrics_test].plot(kind="bar", ax=axes[0], colormap="tab10", width=0.8)
axes[0].set_title("Test Metrics Comparison")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].legend(loc="lower right")
axes[0].tick_params(axis="x", rotation=45)

results_df[["Train Accuracy", "Test Accuracy"]].plot(
    kind="bar", ax=axes[1], color=["steelblue", "coral"], width=0.6)
axes[1].set_title("Train vs Test Accuracy")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=150)
plt.show()



# ─────────────────────────────────────────────
# 9. DONE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print("Files saved:")
print("  classification_results.csv")
print("  results_comparison.png")
print("  confusion_matrix_<model>.png  (one per model)")