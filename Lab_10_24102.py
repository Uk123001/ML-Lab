"""
Galaxy10 DECaLS - Lab Session 10
"""

# Imported Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# no GUI popups, save directly
matplotlib.use('Agg')  
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import lime
import lime.lime_tabular
import shap

# CONFIGURATIONS
CSV_PATH = "galaxy_cnn_features.csv"

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "Inbetween Round Smooth",
    "Cigar Shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "EdgeOn No Bulge", "EdgeOn With Bulge"
]

# LOAD DATA
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

cols_to_drop = [c for c in ["label", "class_name"] if c in df.columns]
X = df.drop(columns=cols_to_drop).values
y = df["label"].values
feature_names = df.drop(columns=cols_to_drop).columns.tolist()

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw)
X_test_sc  = scaler.transform(X_test_raw)

print(f"Train: {X_train_sc.shape[0]} | Test: {X_test_sc.shape[0]}")

# RECREATE PCA-99 (for A5)
print("\nPreparing PCA-99 features for A5...")
pca_99 = PCA(n_components=0.99, random_state=42)
X_train_pca99 = pca_99.fit_transform(X_train_sc)
X_test_pca99  = pca_99.transform(X_test_sc)
n_components_99 = pca_99.n_components_
print(f"  PCA-99: {n_components_99} components retained")

# HELPERS
def get_classifiers():
    """Returns all 8 classifiers from Lab 8 with best hyperparameters."""
    return [
        ("SVM",          SVC(kernel="rbf", C=1, gamma="auto", probability=True, random_state=42)),
        ("Decision Tree",DecisionTreeClassifier(max_depth=10, min_samples_split=10,
                                                criterion="gini", random_state=42)),
        ("Random Forest",RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                                 random_state=42, n_jobs=-1)),
        ("CatBoost",     CatBoostClassifier(iterations=100, learning_rate=0.1,
                                             depth=6, verbose=0, random_state=42)),
        ("AdaBoost",     AdaBoostClassifier(n_estimators=100, learning_rate=1.0,
                                             random_state=42)),
        ("XGBoost",      XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                        subsample=0.9, eval_metric="mlogloss",
                                        verbosity=0, random_state=42)),
        ("Naive Bayes",  GaussianNB(var_smoothing=1e-9)),
        ("MLP",          MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                                        learning_rate_init=0.001, max_iter=300,
                                        random_state=42)),
    ]


def evaluate_all(X_tr, X_te, y_tr, y_te, tag=""):
    """Train and evaluate all 8 classifiers. Returns results DataFrame."""
    results = []
    for name, clf in get_classifiers():
        print(f"  [{tag}] Training {name}...")
        clf.fit(X_tr, y_tr)
        tr_pred = clf.predict(X_tr)
        te_pred = clf.predict(X_te)
        results.append({
            "Model"         : name,
            "Train Acc"     : round(accuracy_score(y_tr, tr_pred), 4),
            "Test Acc"      : round(accuracy_score(y_te, te_pred), 4),
            "Test Precision": round(precision_score(y_te, te_pred, average="macro", zero_division=0), 4),
            "Test Recall"   : round(recall_score(y_te, te_pred, average="macro", zero_division=0), 4),
            "Test F1"       : round(f1_score(y_te, te_pred, average="macro", zero_division=0), 4),
        })
    return pd.DataFrame(results).set_index("Model")


def plot_results_bar(df, title, fname):
    df[["Train Acc", "Test Acc", "Test Precision", "Test Recall", "Test F1"]].plot(
        kind="bar", figsize=(13, 5), colormap="tab10", width=0.75)
    plt.title(title)
    plt.ylabel("Score"); plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

# A1: FEATURE CORRELATION ANALYSIS
print("\n" + "="*60)
print("A1: Feature Correlation Analysis")
print("="*60)

# With 512 features a full heatmap is unreadable — randomly sample 100 features
# This is standard practice and noted in the report
N_SAMPLE = 100
np.random.seed(42)
sampled_indices = np.random.choice(X_train_sc.shape[1], size=N_SAMPLE, replace=False)
sampled_indices.sort()
feat_df = pd.DataFrame(X_train_sc[:, sampled_indices],
                       columns=[feature_names[i] for i in sampled_indices])
corr_matrix = feat_df.corr()

plt.figure(figsize=(20, 18))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.1,
    annot=False,
    xticklabels=False,
    yticklabels=False,
)
plt.title(f"A1: Feature Correlation Heatmap (Random {N_SAMPLE} CNN Features)", fontsize=14)
plt.tight_layout()
plt.savefig("a1_correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved a1_correlation_heatmap.png")

# Summary statistics
mean_abs_corr = corr_matrix.abs().values[np.triu_indices_from(corr_matrix, k=1)].mean()
high_corr_pairs = (corr_matrix.abs() > 0.8).sum().sum() // 2
print(f"  Mean absolute correlation (off-diagonal): {mean_abs_corr:.4f}")
print(f"  Feature pairs with |corr| > 0.8: {high_corr_pairs}")
print("  Observation: CNN features from ResNet18's avgpool layer tend to be "
      "moderately correlated, justifying dimensionality reduction via PCA.")

# A2: PCA — 99% EXPLAINED VARIANCE
print("\n" + "="*60)
print("A2: PCA (99% explained variance) + All 8 Classifiers")
print("="*60)

pca_99 = PCA(n_components=0.99, random_state=42)
X_train_pca99 = pca_99.fit_transform(X_train_sc)
X_test_pca99  = pca_99.transform(X_test_sc)

n_components_99 = pca_99.n_components_
print(f"  Components to retain 99% variance: {n_components_99} (from {X_train_sc.shape[1]})")

# Cumulative variance plot
cumvar = np.cumsum(pca_99.explained_variance_ratio_)
plt.figure(figsize=(9, 4))
plt.plot(range(1, len(cumvar)+1), cumvar, color="steelblue", linewidth=1.5)
plt.axhline(0.99, color="red", linestyle="--", label="99% threshold")
plt.axvline(n_components_99, color="orange", linestyle="--",
            label=f"{n_components_99} components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("A2: PCA Cumulative Explained Variance (99%)")
plt.legend(); plt.tight_layout()
plt.savefig("a2_pca99_cumvar.png", dpi=150)
plt.close()
print("Saved a2_pca99_cumvar.png")

results_pca99 = evaluate_all(X_train_pca99, X_test_pca99, y_train, y_test, tag="PCA-99%")
print("\nA2 Results (PCA 99%):")
print(results_pca99.to_string())
results_pca99.to_csv("a2_pca99_results.csv")
plot_results_bar(results_pca99,
                 f"A2: All Classifiers with PCA 99% ({n_components_99} components)",
                 "a2_pca99_comparison.png")

# A3: PCA — 95% EXPLAINED VARIANCE
print("\n" + "="*60)
print("A3: PCA (95% explained variance) + All 8 Classifiers")
print("="*60)

pca_95 = PCA(n_components=0.95, random_state=42)
X_train_pca95 = pca_95.fit_transform(X_train_sc)
X_test_pca95  = pca_95.transform(X_test_sc)

n_components_95 = pca_95.n_components_
print(f"  Components to retain 95% variance: {n_components_95} (from {X_train_sc.shape[1]})")

# Cumulative variance plot
cumvar95 = np.cumsum(pca_95.explained_variance_ratio_)
plt.figure(figsize=(9, 4))
plt.plot(range(1, len(cumvar95)+1), cumvar95, color="darkorange", linewidth=1.5)
plt.axhline(0.95, color="red", linestyle="--", label="95% threshold")
plt.axvline(n_components_95, color="steelblue", linestyle="--",
            label=f"{n_components_95} components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("A3: PCA Cumulative Explained Variance (95%)")
plt.legend(); plt.tight_layout()
plt.savefig("a3_pca95_cumvar.png", dpi=150)
plt.close()
print("Saved a3_pca95_cumvar.png")

results_pca95 = evaluate_all(X_train_pca95, X_test_pca95, y_train, y_test, tag="PCA-95%")
print("\nA3 Results (PCA 95%):")
print(results_pca95.to_string())
results_pca95.to_csv("a3_pca95_results.csv")
plot_results_bar(results_pca95,
                 f"A3: All Classifiers with PCA 95% ({n_components_95} components)",
                 "a3_pca95_comparison.png")

# COMPARISON PLOT: Baseline vs PCA99 vs PCA95
print("\nGenerating PCA comparison plot...")

# Run baseline (full features) for comparison
print("  Running baseline (full 512 features) for comparison...")
results_baseline = evaluate_all(X_train_sc, X_test_sc, y_train, y_test, tag="Baseline")
results_baseline.to_csv("baseline_results.csv")

# Test accuracy comparison across all three settings
compare_df = pd.DataFrame({
    "Baseline (512)"        : results_baseline["Test Acc"],
    f"PCA 99% ({n_components_99})" : results_pca99["Test Acc"],
    f"PCA 95% ({n_components_95})" : results_pca95["Test Acc"],
})

compare_df.plot(kind="bar", figsize=(13, 5), colormap="Set2", width=0.75)
plt.title("A2 vs A3: Test Accuracy — Baseline vs PCA 99% vs PCA 95%")
plt.ylabel("Test Accuracy"); plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("pca_comparison.png", dpi=150)
plt.close()
print("Saved pca_comparison.png")

# A4: RFE FEATURE SELECTION
print("\n" + "="*60)
print("A4: RFE Feature Selection")
print("="*60)
print("  Using Recursive Feature Elimination (RFE) with Random Forest estimator.")
print("  RFE was chosen over sequential forward selection (SFS) because SFS on")
print("  512 CNN features is computationally intractable (O(n^2) evaluations).")
print("  RFE recursively prunes the least important features using RF importances,")
print("  achieving sequential reduction efficiently in O(n*log n) time.")

# Select top N features — we use 50 as it's comparable to PCA-95 component count
N_FEATURES_RFE = 50
print(f"  Selecting top {N_FEATURES_RFE} features via RFE...")

rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=N_FEATURES_RFE, step=20)
rfe.fit(X_train_sc, y_train)

X_train_rfe = rfe.transform(X_train_sc)
X_test_rfe  = rfe.transform(X_test_sc)

selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
print(f"  Selected {len(selected_features)} features.")
print(f"  First 10 selected: {selected_features[:10]}")

# Feature ranking plot — top 50 by rank
ranking = pd.Series(rfe.ranking_, index=feature_names)
top_ranked = ranking[rfe.support_].sort_values()
plt.figure(figsize=(12, 5))
top_ranked.head(30).plot(kind="bar", color="steelblue")
plt.title(f"A4: RFE — Top 30 Selected Features (of {N_FEATURES_RFE} total selected)")
plt.ylabel("RFE Ranking (lower = more important)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("a4_rfe_features.png", dpi=150)
plt.close()
print("Saved a4_rfe_features.png")

results_rfe = evaluate_all(X_train_rfe, X_test_rfe, y_train, y_test, tag="RFE")
print("\nA4 Results (RFE):")
print(results_rfe.to_string())
results_rfe.to_csv("a4_rfe_results.csv")
plot_results_bar(results_rfe,
                 f"A4: All Classifiers with RFE ({N_FEATURES_RFE} features)",
                 "a4_rfe_comparison.png")

# Full comparison: Baseline vs PCA99 vs PCA95 vs RFE
compare_all = pd.DataFrame({
    "Baseline (512)"              : results_baseline["Test Acc"],
    f"PCA 99% ({n_components_99})": results_pca99["Test Acc"],
    f"PCA 95% ({n_components_95})": results_pca95["Test Acc"],
    f"RFE ({N_FEATURES_RFE})"     : results_rfe["Test Acc"],
})
compare_all.plot(kind="bar", figsize=(14, 5), colormap="tab10", width=0.75)
plt.title("A4: Test Accuracy — Baseline vs PCA 99% vs PCA 95% vs RFE")
plt.ylabel("Test Accuracy"); plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("a4_full_comparison.png", dpi=150)
plt.close()
print("Saved a4_full_comparison.png")

# A5: LIME + SHAP EXPLAINABILITY
print("\n" + "="*60)
print("A5: LIME and SHAP Explainability")
print("="*60)

# Use XGBoost trained on PCA-99 features for explainability
print("  Training XGBoost on PCA-99 features for explainability...")
xgb_explain = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                              subsample=0.9, eval_metric="mlogloss",
                              verbosity=0, random_state=42)
xgb_explain.fit(X_train_pca99, y_train)

pca_feature_names = [f"PC{i+1}" for i in range(n_components_99)]

# LIME 
print("\n  Running LIME...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data  = X_train_pca99,
    feature_names  = pca_feature_names,
    class_names    = CLASS_NAMES,
    mode           = "classification",
    random_state   = 42,
)

for sample_idx in [0, 1, 2]:
    sample     = X_test_pca99[sample_idx]
    true_label = y_test[sample_idx]
    pred_label = xgb_explain.predict([sample])[0]

    exp = lime_explainer.explain_instance(
        data_row   = sample,
        predict_fn = xgb_explain.predict_proba,
        num_features = 15,
        top_labels = 1,
    )

    print(f"\n  LIME Sample {sample_idx}: "
          f"True={CLASS_NAMES[true_label]} | Pred={CLASS_NAMES[pred_label]}")
    for feat, weight in exp.as_list(label=exp.top_labels[0])[:5]:
        direction = "↑ supports" if weight > 0 else "↓ opposes"
        print(f"    {direction} | {feat} (weight={weight:.4f})")

    fig = exp.as_pyplot_figure(label=exp.top_labels[0])
    plt.title(f"A5 LIME | Sample {sample_idx} | "
              f"True: {CLASS_NAMES[true_label]} | Pred: {CLASS_NAMES[pred_label]}")
    plt.tight_layout()
    fname = f"a5_lime_sample{sample_idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")

# SHAP
print("\n  Running SHAP (TreeExplainer on XGBoost)...")
shap_explainer = shap.TreeExplainer(xgb_explain)

# Use 200 test samples for speed
X_shap = X_test_pca99[:200]
shap_values = shap_explainer.shap_values(X_shap)

# Handle both old and new SHAP output formats 
# Old SHAP (<0.40):  returns list of (n_samples, n_features) → one per class
# New SHAP (>=0.40): returns ndarray of shape (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    # Old format
    shap_arr        = np.array(shap_values)          # (n_classes, n_samples, n_features)
    shap_mean       = np.abs(shap_arr).mean(axis=0)  # (n_samples, n_features)
    shap_class0     = shap_arr[0]                    # (n_samples, n_features)
    sv_single       = shap_arr[0][0]                 # (n_features,)
    ev              = shap_explainer.expected_value
    expected_value_0 = ev[0] if isinstance(ev, (list, np.ndarray)) else ev
else:
    # New format: (n_samples, n_features, n_classes)
    shap_arr        = shap_values
    shap_mean       = np.abs(shap_arr).mean(axis=2)  # (n_samples, n_features)
    shap_class0     = shap_arr[:, :, 0]              # (n_samples, n_features)
    sv_single       = shap_arr[0, :, 0]              # (n_features,)
    ev              = shap_explainer.expected_value
    expected_value_0 = ev[0] if isinstance(ev, (list, np.ndarray)) else ev

print(f"  SHAP raw values shape: {shap_arr.shape}")
print(f"  X_shap shape: {X_shap.shape}")

# Global summary bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_mean,
    X_shap,
    feature_names=pca_feature_names,
    plot_type="bar",
    show=False,
)
plt.title("A5: SHAP Global Feature Importance (XGBoost on PCA-99 features)")
plt.tight_layout()
plt.savefig("a5_shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved a5_shap_summary_bar.png")

# Beeswarm / dot plot for class 0 (Disturbed)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_class0,
    X_shap,
    feature_names=pca_feature_names,
    plot_type="dot",
    show=False,
)
plt.title("A5: SHAP Beeswarm — Class 0 (Disturbed)")
plt.tight_layout()
plt.savefig("a5_shap_beeswarm_class0.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved a5_shap_beeswarm_class0.png")

# SHAP waterfall plot for sample 0, class 0
shap.initjs()
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values        = sv_single,
        base_values   = expected_value_0,
        data          = X_shap[0],
        feature_names = pca_feature_names,
    ),
    show=False,
)
plt.title("A5: SHAP Waterfall — Sample 0 (Class: Disturbed)")
plt.tight_layout()
plt.savefig("a5_shap_waterfall_sample0.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved a5_shap_waterfall_sample0.png")

# LIME vs SHAP Comparison Summary
print("\n  LIME vs SHAP Comparison:")
print("  ┌─────────────┬──────────────────────────────┬──────────────────────────────┐")
print("  │ Property    │ LIME                         │ SHAP                         │")
print("  ├─────────────┼──────────────────────────────┼──────────────────────────────┤")
print("  │ Scope       │ Local (per instance)         │ Local + Global               │")
print("  │ Method      │ Local linear approximation   │ Shapley values (game theory) │")
print("  │ Consistency │ Can vary across runs         │ Theoretically consistent     │")
print("  │ Speed       │ Fast                         │ Fast for tree models         │")
print("  │ Best for    │ Explaining single predictions│ Global feature importance    │")
print("  └─────────────┴──────────────────────────────┴──────────────────────────────┘")

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print("Files saved:")
print("  a5_lime_sample0.png, a5_lime_sample1.png, a5_lime_sample2.png")
print("  a5_shap_summary_bar.png")
print("  a5_shap_beeswarm_class0.png")
print("  a5_shap_waterfall_sample0.png")