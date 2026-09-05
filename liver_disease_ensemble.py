"""
Liver Disease Classification - Ensemble Models
==================================================
Combines the three strongest individual classifiers (tuned Logistic Regression,
Random Forest, SVM from the advanced pipeline) into ensembles, to see whether
combining models beats any single one:

    1. Soft Voting Classifier  - averages predicted probabilities
    2. Stacking Classifier     - trains a meta-model (Logistic Regression) on
                                  the base models' predictions

Both are trained on SMOTE-balanced data using each base model's tuned
hyperparameters (loaded from outputs_advanced/best_hyperparameters.txt logic,
re-specified here directly) and evaluated on the same held-out test set used
throughout the project, for a fair, apples-to-apples comparison.

Run:
    python liver_disease_ensemble.py
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH = "data/indian_liver_patient.csv"
OUTPUT_DIR = "outputs_ensemble"
RANDOM_STATE = 42

COLUMN_NAMES = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Proteins", "Albumin",
    "Albumin_and_Globulin_Ratio", "Dataset"
]


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
    print("\nLIVER DISEASE CLASSIFICATION - ENSEMBLE MODELS\n")

    # 1. Load, preprocess, split (same split used throughout the project)
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    # 2. SMOTE on training data only (same approach as the advanced pipeline)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    # 3. Base learners, using the tuned hyperparameters found by GridSearchCV
    #    in liver_disease_classification_advanced.py
    base_learners = [
        ("logreg", LogisticRegression(C=0.1, max_iter=3000, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=RANDOM_STATE
        )),
        ("svm", SVC(C=50, gamma="scale", kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # 4. Individual tuned models (for comparison, refit on SMOTE data)
    print("Training individual tuned base models (on SMOTE-balanced data)...")
    for name, model in base_learners:
        model.fit(X_train_sm, y_train_sm)
        metrics, y_pred, y_proba = evaluate(model, X_test_scaled, y_test)
        results[name] = {"model": model, "metrics": metrics, "y_pred": y_pred, "y_proba": y_proba}
        print(f"  {name:10s} | Acc: {metrics['Test Accuracy']:.3f} | "
              f"F1: {metrics['Test F1']:.3f} | ROC-AUC: {metrics['Test ROC-AUC']:.3f}")

    # 5. Soft Voting Classifier
    print("\nTraining Soft Voting Classifier...")
    voting = VotingClassifier(estimators=base_learners, voting="soft")
    voting.fit(X_train_sm, y_train_sm)
    v_metrics, v_pred, v_proba = evaluate(voting, X_test_scaled, y_test)
    v_cv = cross_val_score(voting, X_train_sm, y_train_sm, cv=cv, scoring="roc_auc")
    results["Voting Ensemble"] = {"model": voting, "metrics": v_metrics, "y_pred": v_pred, "y_proba": v_proba}
    print(f"  Voting     | Acc: {v_metrics['Test Accuracy']:.3f} | "
          f"F1: {v_metrics['Test F1']:.3f} | ROC-AUC: {v_metrics['Test ROC-AUC']:.3f} "
          f"| CV ROC-AUC: {v_cv.mean():.3f}")

    # 6. Stacking Classifier (meta-learner: Logistic Regression on base model outputs)
    print("\nTraining Stacking Classifier...")
    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=cv, stack_method="predict_proba"
    )
    stacking.fit(X_train_sm, y_train_sm)
    s_metrics, s_pred, s_proba = evaluate(stacking, X_test_scaled, y_test)
    s_cv = cross_val_score(stacking, X_train_sm, y_train_sm, cv=cv, scoring="roc_auc")
    results["Stacking Ensemble"] = {"model": stacking, "metrics": s_metrics, "y_pred": s_pred, "y_proba": s_proba}
    print(f"  Stacking   | Acc: {s_metrics['Test Accuracy']:.3f} | "
          f"F1: {s_metrics['Test F1']:.3f} | ROC-AUC: {s_metrics['Test ROC-AUC']:.3f} "
          f"| CV ROC-AUC: {s_cv.mean():.3f}")

    # 7. Full comparison table
    rows = []
    display_names = {"logreg": "Logistic Regression", "rf": "Random Forest", "svm": "SVM (RBF)",
                      "Voting Ensemble": "Voting Ensemble", "Stacking Ensemble": "Stacking Ensemble"}
    for key, res in results.items():
        rows.append({"Model": display_names[key], **res["metrics"]})
    comparison_df = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False)
    comparison_df.to_csv(f"{OUTPUT_DIR}/ensemble_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("FULL COMPARISON: INDIVIDUAL MODELS vs ENSEMBLES")
    print("=" * 70)
    print(comparison_df.round(3).to_string(index=False))

    best_name = comparison_df.iloc[0]["Model"]
    best_key = [k for k, v in display_names.items() if v == best_name][0]
    print(f"\nBest overall: {best_name}")
    print("\nClassification report (best model):")
    print(classification_report(y_test, results[best_key]["y_pred"],
                                 target_names=["No Disease", "Disease"]))

    # 8. Plots
    plt.figure(figsize=(8, 5))
    order = comparison_df["Model"]
    colors = ["#e67e22" if "Ensemble" in m else "#3498db" for m in order]
    plt.bar(order, comparison_df["Test ROC-AUC"], color=colors)
    plt.ylabel("Test ROC-AUC")
    plt.title("Individual Models vs Ensembles (Test ROC-AUC)")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ensemble_vs_individual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    for key, res in results.items():
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            style = "-" if "Ensemble" in display_names[key] else "--"
            lw = 2.5 if "Ensemble" in display_names[key] else 1.5
            plt.plot(fpr, tpr, style, linewidth=lw,
                     label=f"{display_names[key]} (AUC={res['metrics']['Test ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k:", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Individual Models (dashed) vs Ensembles (solid)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ensemble_roc_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(4.5, 4))
    cm = confusion_matrix(y_test, results[best_key]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/best_ensemble_confusion_matrix.png", dpi=150)
    plt.close()

    # 9. Save best model + scaler + a small SHAP background sample (for downstream
    #    model-agnostic explanations via KernelExplainer, since the winning model
    #    may be a Voting/Stacking ensemble rather than a single tree model)
    joblib.dump(results[best_key]["model"],
                f"{OUTPUT_DIR}/best_model_{best_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.joblib")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")

    import shap
    background = shap.kmeans(X_train_scaled, 25)
    joblib.dump(background, f"{OUTPUT_DIR}/shap_background.joblib")

    print(f"\nAll outputs written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
