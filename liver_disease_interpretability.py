"""
Liver Disease Classification - Model Interpretability (SHAP)
================================================================
Explains *why* the model predicts liver disease for a given patient,
using SHAP (SHapley Additive exPlanations) on the tuned Random Forest
from the advanced pipeline (tree-based models give exact, fast SHAP values
via TreeExplainer -- more reliable than KernelExplainer on RBF-SVM/LogReg).

Produces:
    - Global feature importance (mean |SHAP value| across the test set)
    - Summary "beeswarm" plot (how each feature pushes predictions up/down)
    - Individual waterfall explanations for a few example patients
      (one correctly-flagged disease case, one correctly-cleared case,
       and the single most confident misclassification, if any)

Run:
    python liver_disease_interpretability.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/indian_liver_patient.csv"
OUTPUT_DIR = "outputs_interpretability"
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nLIVER DISEASE CLASSIFICATION - SHAP INTERPRETABILITY\n")

    # 1. Load, preprocess, split -- same split as the other scripts (random_state=42)
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    # 2. Train a Random Forest (tree explainer is exact + fast; RF was competitive
    #    in both the baseline and tuned comparisons)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"Random Forest test accuracy: {(y_pred == y_test).mean():.3f}")

    # 3. SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # shap_values shape handling: newer SHAP returns array (n, features, classes) for
    # multiclass/binary RF; take the "disease" (class 1) slice either way.
    if isinstance(shap_values, list):
        sv_disease = shap_values[1]
    elif shap_values.ndim == 3:
        sv_disease = shap_values[:, :, 1]
    else:
        sv_disease = shap_values

    # 4. Global feature importance (mean |SHAP value|)
    mean_abs_shap = pd.Series(
        np.abs(sv_disease).mean(axis=0), index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(7, 5))
    mean_abs_shap.plot(kind="barh", color="#8e44ad")
    plt.title("Global Feature Importance (mean |SHAP value|)")
    plt.xlabel("Mean |SHAP value| (impact on model output)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_global_importance.png", dpi=150)
    plt.close()

    print("\nTop 5 features by mean |SHAP value|:")
    print(mean_abs_shap.sort_values(ascending=False).head(5).round(4).to_string())

    # 5. Beeswarm summary plot (shows direction + magnitude per feature)
    plt.figure()
    shap.summary_plot(sv_disease, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Individual explanations: one true positive, one true negative,
    #    and the most confident misclassification (if any exist)
    test_idx = X_test_scaled.reset_index(drop=True).index
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_reset = pd.Series(y_pred)
    y_proba_reset = pd.Series(y_proba)

    correct_disease = test_idx[(y_test_reset == 1) & (y_pred_reset == 1)]
    correct_healthy = test_idx[(y_test_reset == 0) & (y_pred_reset == 0)]
    misclassified = test_idx[y_test_reset != y_pred_reset]

    examples = {}
    if len(correct_disease) > 0:
        # most confident correct disease prediction
        i = correct_disease[y_proba_reset[correct_disease].values.argmax()]
        examples["confident_correct_disease_case"] = i
    if len(correct_healthy) > 0:
        i = correct_healthy[y_proba_reset[correct_healthy].values.argmin()]
        examples["confident_correct_no_disease_case"] = i
    if len(misclassified) > 0:
        # most confident WRONG prediction -- i.e. where the model was most surprised to be wrong
        dist_from_0_5 = (y_proba_reset[misclassified] - 0.5).abs()
        i = misclassified[dist_from_0_5.values.argmax()]
        examples["most_confident_misclassification"] = i

    print(f"\nGenerating individual waterfall explanations for {len(examples)} example patients...")
    explanation = shap.Explanation(
        values=sv_disease,
        base_values=np.full(len(sv_disease), explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
        data=X_test_scaled.values,
        feature_names=feature_names,
    )

    for label, i in examples.items():
        plt.figure()
        shap.plots.waterfall(explanation[i], show=False, max_display=10)
        plt.title(
            f"{label}\nActual: {'Disease' if y_test_reset[i] == 1 else 'No Disease'} | "
            f"Predicted: {'Disease' if y_pred_reset[i] == 1 else 'No Disease'} "
            f"(P={y_proba_reset[i]:.2f})",
            fontsize=9
        )
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 7. Save the raw SHAP values + model for further analysis
    joblib.dump(model, f"{OUTPUT_DIR}/rf_model_for_shap.joblib")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")
    np.save(f"{OUTPUT_DIR}/shap_values_test_set.npy", sv_disease)

    print(f"\nAll outputs written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
