"""
Liver Disease Classification - Advanced Pipeline
===================================================
Extends the baseline pipeline (liver_disease_classification.py) with:
    1. SMOTE oversampling to address class imbalance (71% disease / 29% no-disease)
    2. GridSearchCV hyperparameter tuning for the top-performing model families
    3. A side-by-side comparison: baseline (default params, no SMOTE) vs tuned (+SMOTE)

Run:
    python liver_disease_classification_advanced.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH = "data/indian_liver_patient.csv"
OUTPUT_DIR = "outputs_advanced"
RANDOM_STATE = 42

COLUMN_NAMES = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Proteins", "Albumin",
    "Albumin_and_Globulin_Ratio", "Dataset"
]

# Search grids for the three model families that led the baseline comparison
PARAM_GRIDS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "SVM (RBF)": {
        "estimator": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "params": {
            "C": [0.1, 1, 10, 50],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
    },
}


def load_data(path):
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def preprocess(df):
    df = df.copy()
    df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    numeric_cols = df.columns.drop(["Gender", "Dataset"])
    df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
    X = df.drop(columns=["Dataset"])
    y = df["Dataset"]
    return X, y


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred, zero_division=0),
        "Test Recall": recall_score(y_test, y_pred, zero_division=0),
        "Test F1": f1_score(y_test, y_pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }, y_pred, y_proba


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nLIVER DISEASE CLASSIFICATION - ADVANCED PIPELINE (SMOTE + GridSearchCV)\n")

    # 1. Load & preprocess
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    print(f"Training set before SMOTE: {y_train.value_counts().to_dict()}")

    # 2. SMOTE oversampling (training set only -- test set stays untouched/real-world)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    print(f"Training set after SMOTE:  {pd.Series(y_train_sm).value_counts().to_dict()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 3. Baseline (default hyperparams, no SMOTE) for comparison
    baseline_results = {}
    for name, cfg in PARAM_GRIDS.items():
        model = cfg["estimator"]
        model.fit(X_train_scaled, y_train)
        metrics, y_pred, y_proba = evaluate(model, X_test_scaled, y_test)
        baseline_results[name] = metrics

    # 4. GridSearchCV tuning, trained on SMOTE-balanced data
    tuned_results = {}
    tuned_models = {}
    print("\n" + "=" * 70)
    print("GRID SEARCH (5-fold CV, scoring=roc_auc, trained on SMOTE-balanced data)")
    print("=" * 70)
    for name, cfg in PARAM_GRIDS.items():
        grid = GridSearchCV(
            cfg["estimator"], cfg["params"], scoring="roc_auc",
            cv=cv, n_jobs=-1, refit=True
        )
        grid.fit(X_train_sm, y_train_sm)
        best_model = grid.best_estimator_
        metrics, y_pred, y_proba = evaluate(best_model, X_test_scaled, y_test)
        tuned_results[name] = {"metrics": metrics, "y_pred": y_pred, "y_proba": y_proba,
                                "best_params": grid.best_params_, "cv_roc_auc": grid.best_score_}
        tuned_models[name] = best_model
        print(f"\n{name}")
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV ROC-AUC: {grid.best_score_:.3f}")
        print(f"  Test set -> Acc: {metrics['Test Accuracy']:.3f} | "
              f"F1: {metrics['Test F1']:.3f} | ROC-AUC: {metrics['Test ROC-AUC']:.3f}")

    # 5. Comparison table: baseline vs tuned+SMOTE
    rows = []
    for name in PARAM_GRIDS:
        b = baseline_results[name]
        t = tuned_results[name]["metrics"]
        rows.append({
            "Model": name,
            "Baseline Acc": b["Test Accuracy"], "Tuned+SMOTE Acc": t["Test Accuracy"],
            "Baseline F1": b["Test F1"], "Tuned+SMOTE F1": t["Test F1"],
            "Baseline ROC-AUC": b["Test ROC-AUC"], "Tuned+SMOTE ROC-AUC": t["Test ROC-AUC"],
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(f"{OUTPUT_DIR}/baseline_vs_tuned_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("BASELINE vs TUNED+SMOTE COMPARISON")
    print("=" * 70)
    print(comparison_df.round(3).to_string(index=False))

    # 6. Pick overall best tuned model by test ROC-AUC
    best_name = max(tuned_results, key=lambda n: tuned_results[n]["metrics"]["Test ROC-AUC"])
    best_model = tuned_models[best_name]
    best_metrics = tuned_results[best_name]["metrics"]
    print(f"\nBest tuned model: {best_name} (Test ROC-AUC={best_metrics['Test ROC-AUC']:.3f})")
    print("\nClassification report (best tuned model):")
    print(classification_report(y_test, tuned_results[best_name]["y_pred"],
                                 target_names=["No Disease", "Disease"]))

    # 7. Plots: grouped bar chart of baseline vs tuned metrics
    metrics_to_plot = ["Acc", "F1", "ROC-AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(PARAM_GRIDS))
    width = 0.35
    for ax, metric in zip(axes, metrics_to_plot):
        baseline_vals = comparison_df[f"Baseline {metric}"]
        tuned_vals = comparison_df[f"Tuned+SMOTE {metric}"]
        ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#95a5a6")
        ax.bar(x + width / 2, tuned_vals, width, label="Tuned + SMOTE", color="#2980b9")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["Model"], rotation=15, ha="right")
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/baseline_vs_tuned.png", dpi=150)
    plt.close()

    # Confusion matrix for best tuned model
    plt.figure(figsize=(4.5, 4))
    cm = confusion_matrix(y_test, tuned_results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - Tuned {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/best_tuned_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curves for all tuned models
    plt.figure(figsize=(6.5, 5.5))
    for name, res in tuned_results.items():
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['Test ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Tuned Models (SMOTE-trained)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tuned_roc_curves.png", dpi=150)
    plt.close()

    # 8. Save best tuned model + scaler
    joblib.dump(best_model, f"{OUTPUT_DIR}/best_tuned_model_{best_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.joblib")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")

    # Save best params for every model as a small reference file
    with open(f"{OUTPUT_DIR}/best_hyperparameters.txt", "w") as f:
        for name, res in tuned_results.items():
            f.write(f"{name}:\n  {res['best_params']}\n  CV ROC-AUC: {res['cv_roc_auc']:.4f}\n\n")

    print(f"\nAll outputs written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
