"""
Galaxy10 DECaLS - Lab Session 9
================================
A1: Stacking Classifier with multiple meta-models
A2: Pipeline (preprocessing + classification)
A3: LIME explainability on Pipeline
"""

# Necessary imported modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Base models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               StackingClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import lime
import lime.lime_tabular

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading features...")
df = pd.read_csv("Lab 9/galaxy_cnn_features.csv")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

cols_to_drop = [c for c in ["label", "class_name"] if c in df.columns]
X = df.drop(columns=cols_to_drop).values
y = df["label"].values
feature_names = df.drop(columns=cols_to_drop).columns.tolist()

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "Inbetween Round Smooth",
    "Cigar Shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "EdgeOn No Bulge", "EdgeOn With Bulge"
]

print(f"Dataset shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    tr_pred = model.predict(X_tr)
    te_pred = model.predict(X_te)
    return {
        "Model"         : name,
        "Train Acc"     : round(accuracy_score(y_tr, tr_pred), 4),
        "Test Acc"      : round(accuracy_score(y_te, te_pred), 4),
        "Test Precision": round(precision_score(y_te, te_pred, average="macro", zero_division=0), 4),
        "Test Recall"   : round(recall_score(y_te, te_pred, average="macro", zero_division=0), 4),
        "Test F1"       : round(f1_score(y_te, te_pred, average="macro", zero_division=0), 4),
    }, te_pred


def plot_cm(y_te, y_pred, name):
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = f"cm_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved {fname}")


# ─────────────────────────────────────────────
# A1: STACKING CLASSIFIER
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("A1: Stacking Classifier")
print("="*60)

# Scale inside stacking manually (base models need scaled data)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Base estimators — all 8 classifiers from Lab 8, SVM last (slowest)
base_estimators = [
    ("dt",   DecisionTreeClassifier(max_depth=10, min_samples_split=10,
                                     criterion="gini", random_state=42)),
    ("rf",   RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                     random_state=42, n_jobs=-1)),
    ("cat",  CatBoostClassifier(iterations=100, learning_rate=0.1,
                                  depth=6, verbose=0, random_state=42)),
    ("ada",  AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)),
    ("xgb",  XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             subsample=0.9, eval_metric="mlogloss",
                             verbosity=0, random_state=42)),
    ("nb",   GaussianNB(var_smoothing=1e-9)),
    ("mlp",  MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                            learning_rate_init=0.001, max_iter=100, random_state=42)),
    ("svm",  SVC(kernel="rbf", C=1, gamma="auto", probability=True, random_state=42)),
]

# Experiment with 3 meta-models
meta_models = {
    "Logistic Regression (meta)": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest (meta)"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (meta)"            : XGBClassifier(n_estimators=100, eval_metric="mlogloss",
                                                  verbosity=0, random_state=42),
}

stacking_results = []
best_stacking_model = None
best_stacking_acc   = 0.0

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

for meta_name, meta_model in meta_models.items():
    print(f"\nStacking with meta-model: {meta_name}")
    print("  Fitting base models across CV folds (SVM will be slow)...")

    stack = StackingClassifier(
        estimators       = base_estimators,
        final_estimator  = meta_model,
        cv               = cv,
        passthrough      = False,
        n_jobs           = 1,
    )

    stack.fit(X_train_sc, y_train)

    tr_pred = stack.predict(X_train_sc)
    te_pred = stack.predict(X_test_sc)

    res = {
        "Meta-Model"    : meta_name,
        "Train Acc"     : round(accuracy_score(y_train, tr_pred), 4),
        "Test Acc"      : round(accuracy_score(y_test,  te_pred), 4),
        "Test Precision": round(precision_score(y_test, te_pred, average="macro", zero_division=0), 4),
        "Test Recall"   : round(recall_score(y_test,    te_pred, average="macro", zero_division=0), 4),
        "Test F1"       : round(f1_score(y_test,        te_pred, average="macro", zero_division=0), 4),
    }
    stacking_results.append(res)

    print(f"  Train Acc : {res['Train Acc']}  |  Test Acc : {res['Test Acc']}")
    print(f"  Precision : {res['Test Precision']}  |  Recall : {res['Test Recall']}  |  F1 : {res['Test F1']}")

    plot_cm(y_test, te_pred, f"Stacking_{meta_name.split()[0]}")

    if res["Test Acc"] > best_stacking_acc:
        best_stacking_acc   = res["Test Acc"]
        best_stacking_model = stack
        best_meta_name      = meta_name

stacking_df = pd.DataFrame(stacking_results).set_index("Meta-Model")
print("\nA1 Stacking Results:")
print(stacking_df.to_string())
stacking_df.to_csv("stacking_results.csv")

# Bar chart for stacking
stacking_df[["Train Acc", "Test Acc", "Test Precision", "Test Recall", "Test F1"]].plot(
    kind="bar", figsize=(10, 5), colormap="tab10", width=0.7)
plt.title("A1: Stacking Classifier — Meta-Model Comparison")
plt.ylabel("Score"); plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("stacking_comparison.png", dpi=150)
plt.show()

print(f"\nBest stacking meta-model: {best_meta_name} (Test Acc = {best_stacking_acc})")

# ─────────────────────────────────────────────
# A2: PIPELINE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("A2: Sklearn Pipeline")
print("="*60)

# Build pipelines for several classifiers to show the concept
# Pipeline handles StandardScaler + classifier in one object
pipeline_configs = [
    ("Pipeline-SVM",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf",    SVC(kernel="rbf", C=1, gamma="auto", probability=True, random_state=42)),
     ])),
    ("Pipeline-RF",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf",    RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                            random_state=42, n_jobs=-1)),
     ])),
    ("Pipeline-XGBoost",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf",    XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                   subsample=0.9, eval_metric="mlogloss",
                                   verbosity=0, random_state=42)),
     ])),
    ("Pipeline-MLP",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf",    MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                                   learning_rate_init=0.001, max_iter=100, random_state=42)),
     ])),
    ("Pipeline-Stacking",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf",    StackingClassifier(
             estimators      = base_estimators,
             final_estimator = meta_models["Logistic Regression (meta)"],
             cv              = cv,
             passthrough     = False,
             n_jobs          = -1,
         )),
     ])),
]

pipeline_results = []
best_pipeline     = None
best_pipeline_acc = 0.0

for pipe_name, pipe in pipeline_configs:
    print(f"\nTraining {pipe_name}...")
    res, te_pred = evaluate(pipe_name, pipe, X_train, X_test, y_train, y_test)
    pipeline_results.append(res)
    print(f"  Train Acc: {res['Train Acc']}  |  Test Acc: {res['Test Acc']}")
    print(f"  F1: {res['Test F1']}")
    plot_cm(y_test, te_pred, pipe_name)

    if res["Test Acc"] > best_pipeline_acc:
        best_pipeline_acc = res["Test Acc"]
        best_pipeline     = pipe
        best_pipeline_name = pipe_name

pipeline_df = pd.DataFrame(pipeline_results).set_index("Model")
print("\nA2 Pipeline Results:")
print(pipeline_df.to_string())
pipeline_df.to_csv("pipeline_results.csv")

pipeline_df[["Train Acc", "Test Acc", "Test Precision", "Test Recall", "Test F1"]].plot(
    kind="bar", figsize=(12, 5), colormap="tab10", width=0.7)
plt.title("A2: Pipeline Classifier Comparison")
plt.ylabel("Score"); plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("pipeline_comparison.png", dpi=150)
plt.show()

print(f"\nBest pipeline: {best_pipeline_name} (Test Acc = {best_pipeline_acc})")

# ─────────────────────────────────────────────
# A3: LIME EXPLAINABILITY ON BEST PIPELINE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("A3: LIME Explainability on Best Pipeline")
print("="*60)

# LIME needs a predict_proba function
# For SVC pipelines that don't expose probability=True at top level:
def pipeline_predict_proba(X_input):
    return best_pipeline.predict_proba(X_input)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = X_train,
    feature_names = feature_names,
    class_names   = CLASS_NAMES,
    mode          = "classification",
    random_state  = 42,
)

# Explain 3 test samples
for sample_idx in [0, 1, 2]:
    sample     = X_test[sample_idx]
    true_label = y_test[sample_idx]
    pred_label = best_pipeline.predict([sample])[0]

    explanation = lime_explainer.explain_instance(
        data_row   = sample,
        predict_fn = pipeline_predict_proba,
        num_features = 15,
        top_labels = 1,
    )

    print(f"\nSample {sample_idx}: True = {CLASS_NAMES[true_label]} | "
          f"Predicted = {CLASS_NAMES[pred_label]}")

    fig = explanation.as_pyplot_figure(label=explanation.top_labels[0])
    plt.title(f"LIME | Sample {sample_idx} | True: {CLASS_NAMES[true_label]} | "
              f"Pred: {CLASS_NAMES[pred_label]}")
    plt.tight_layout()
    fname = f"lime_pipeline_sample{sample_idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")

    # Print top 5 features for this prediction
    top_features = explanation.as_list(label=explanation.top_labels[0])
    print("  Top contributing features:")
    for feat, weight in top_features[:5]:
        direction = "↑ supports" if weight > 0 else "↓ opposes"
        print(f"    {direction} prediction | {feat} (weight={weight:.4f})")

# ─────────────────────────────────────────────
# COMBINED RESULTS TABLE (Stacking + Pipeline)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("COMBINED RESULTS SUMMARY")
print("="*60)

print("\n--- A1: Stacking ---")
print(stacking_df.to_string())
print("\n--- A2: Pipeline ---")
print(pipeline_df.to_string())

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print("Files saved:")
print("  stacking_results.csv")
print("  pipeline_results.csv")
print("  stacking_comparison.png")
print("  pipeline_comparison.png")
print("  cm_stacking_*.png  (one per meta-model)")
print("  cm_pipeline_*.png  (one per pipeline)")
print("  lime_pipeline_sample0.png")
print("  lime_pipeline_sample1.png")
print("  lime_pipeline_sample2.png")
