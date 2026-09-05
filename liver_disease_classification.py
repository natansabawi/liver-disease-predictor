"""
Liver Disease Classification using Machine Learning
=====================================================
Dataset: Indian Liver Patient Dataset (ILPD) - UCI Machine Learning Repository
Task: Binary classification - predict whether a patient has liver disease
      based on clinical/blood-test attributes.

Pipeline:
    1. Load & clean data
    2. Exploratory data analysis (saved as plots)
    3. Preprocessing (encoding, imputation, scaling)
    4. Train multiple classifiers with cross-validation
    5. Evaluate on held-out test set (accuracy, precision, recall, F1, ROC-AUC)
    6. Compare models, save the best one, plot feature importance

Run:
    python liver_disease_classification.py
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH = "data/indian_liver_patient.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

COLUMN_NAMES = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Proteins", "Albumin",
    "Albumin_and_Globulin_Ratio", "Dataset"
]


def load_data(path):
    """Load the ILPD dataset and attach column names."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def explore_data(df):
    """Basic EDA: print summary stats and save a handful of diagnostic plots."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nClass distribution (Dataset column, 1=disease, 2=no disease):")
    print(df["Dataset"].value_counts())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Class balance
    plt.figure(figsize=(5, 4))
    counts = df["Dataset"].map({1: "Liver Disease", 2: "No Disease"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette=["#c0392b", "#27ae60"])
    plt.title("Class Distribution")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
    plt.close()

    # Correlation heatmap (numeric features only)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    # Age distribution by class
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="Age", hue="Dataset", kde=True, palette=["#c0392b", "#27ae60"])
    plt.title("Age Distribution by Diagnosis")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/age_distribution.png", dpi=150)
    plt.close()

    print(f"\nEDA plots saved to '{OUTPUT_DIR}/'")


def preprocess(df):
    """Clean, encode, impute, and split features/target."""
    df = df.copy()

    # Target: convert {1: disease, 2: no disease} -> {1: disease, 0: no disease}
    df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})

    # Encode Gender
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])  # Female=0, Male=1

    # Impute missing numeric values (Albumin_and_Globulin_Ratio has a few NaNs)
    imputer = SimpleImputer(strategy="median")
    numeric_cols = df.columns.drop(["Gender", "Dataset"])
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    X = df.drop(columns=["Dataset"])
    y = df["Dataset"]

    return X, y


def get_models():
    """Return a dict of model_name -> sklearn-compatible estimator."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
        )
    return models


def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv):
    """Fit a model, run cross-validation, and compute test-set metrics."""
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "CV ROC-AUC (mean)": cv_scores.mean(),
        "CV ROC-AUC (std)": cv_scores.std(),
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred, zero_division=0),
        "Test Recall": recall_score(y_test, y_pred, zero_division=0),
        "Test F1": f1_score(y_test, y_pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    return model, metrics, y_pred, y_proba


def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all trained models in a grid."""
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                    xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(results, y_test):
    """Plot overlaid ROC curves for all models that support predict_proba."""
    plt.figure(figsize=(7, 6))
    for name, res in results.items():
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            auc = res["metrics"]["Test ROC-AUC"]
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#2980b9")
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()


def main():
    print("\nLIVER DISEASE CLASSIFICATION USING MACHINE LEARNING\n")

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. EDA
    explore_data(df)

    # 3. Preprocess
    X, y = preprocess(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    # 4. Train & evaluate all models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATING MODELS (5-fold CV + held-out test set)")
    print("=" * 60)
    for name, model in models.items():
        fitted_model, metrics, y_pred, y_proba = evaluate_model(
            name, model, X_train_scaled, X_test_scaled, y_train, y_test, cv
        )
        results[name] = {
            "model": fitted_model, "metrics": metrics,
            "y_pred": y_pred, "y_proba": y_proba
        }
        print(f"{name:22s} | CV ROC-AUC: {metrics['CV ROC-AUC (mean)']:.3f} "
              f"| Test Acc: {metrics['Test Accuracy']:.3f} "
              f"| Test F1: {metrics['Test F1']:.3f} "
              f"| Test ROC-AUC: {metrics['Test ROC-AUC']:.3f}")

    # 5. Comparison table
    comparison_df = pd.DataFrame([r["metrics"] for r in results.values()]).sort_values(
        "Test ROC-AUC", ascending=False
    )
    comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by Test ROC-AUC)")
    print("=" * 60)
    print(comparison_df.round(3).to_string(index=False))

    # 6. Best model
    best_name = comparison_df.iloc[0]["Model"]
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")
    print("\nClassification report (best model):")
    print(classification_report(y_test, results[best_name]["y_pred"],
                                 target_names=["No Disease", "Disease"]))

    # 7. Plots
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)

    # Feature importance: prefer Random Forest if available, else best model if tree-based
    importance_source = results.get("Random Forest", results[best_name])
    plot_feature_importance(importance_source["model"], feature_names,
                             "Random Forest" if "Random Forest" in results else best_name)

    # 8. Save best model + scaler for reuse
    joblib.dump(best_model, f"{OUTPUT_DIR}/best_model_{best_name.replace(' ', '_').lower()}.joblib")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")

    print(f"\nAll outputs (plots, comparison table, saved model) written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
