"""
Liver Disease Classification - Single-Patient Prediction CLI
================================================================
Loads the project's best overall model -- the Soft Voting Ensemble
(tuned Logistic Regression + Random Forest + SVM) -- and predicts liver
disease risk for one patient, with a SHAP-based explanation of the top
contributing factors.

Because the Voting Ensemble mixes model types (linear, tree, kernel), it
isn't compatible with SHAP's fast TreeExplainer, so this script uses
KernelExplainer (model-agnostic) against a small k-means background summary
of the training data -- fast enough for single-patient, on-demand use.

Usage (interactive prompts):
    python predict_patient.py

Usage (all values via flags, for scripting):
    python predict_patient.py --age 45 --gender Male --total_bilirubin 1.2 \\
        --direct_bilirubin 0.4 --alkphos 200 --alamine 30 --aspartate 35 \\
        --total_proteins 7.0 --albumin 3.5 --ag_ratio 1.0

Requires outputs_ensemble/best_model_voting_ensemble.joblib, scaler.joblib,
and shap_background.joblib -- run liver_disease_ensemble.py first if missing.
"""

import argparse
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "outputs_ensemble/best_model_voting_ensemble.joblib"
SCALER_PATH = "outputs_ensemble/scaler.joblib"
BACKGROUND_PATH = "outputs_ensemble/shap_background.joblib"

FEATURE_NAMES = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Proteins", "Albumin",
    "Albumin_and_Globulin_Ratio"
]

PROMPTS = {
    "Age": ("Age (years)", float),
    "Gender": ("Gender (Male/Female)", str),
    "Total_Bilirubin": ("Total Bilirubin (mg/dL)", float),
    "Direct_Bilirubin": ("Direct Bilirubin (mg/dL)", float),
    "Alkaline_Phosphotase": ("Alkaline Phosphotase (IU/L)", float),
    "Alamine_Aminotransferase": ("Alamine Aminotransferase / ALT (IU/L)", float),
    "Aspartate_Aminotransferase": ("Aspartate Aminotransferase / AST (IU/L)", float),
    "Total_Proteins": ("Total Proteins (g/dL)", float),
    "Albumin": ("Albumin (g/dL)", float),
    "Albumin_and_Globulin_Ratio": ("Albumin/Globulin Ratio", float),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict liver disease risk for one patient.")
    parser.add_argument("--age", type=float)
    parser.add_argument("--gender", type=str, choices=["Male", "Female", "male", "female"])
    parser.add_argument("--total_bilirubin", type=float)
    parser.add_argument("--direct_bilirubin", type=float)
    parser.add_argument("--alkphos", type=float, help="Alkaline Phosphotase")
    parser.add_argument("--alamine", type=float, help="Alamine Aminotransferase (ALT)")
    parser.add_argument("--aspartate", type=float, help="Aspartate Aminotransferase (AST)")
    parser.add_argument("--total_proteins", type=float)
    parser.add_argument("--albumin", type=float)
    parser.add_argument("--ag_ratio", type=float, help="Albumin/Globulin Ratio")
    parser.add_argument("--fast", action="store_true",
                         help="Skip the SHAP explanation for a quicker result (prediction only).")
    return parser.parse_args()


def collect_from_args(args):
    mapping = {
        "Age": args.age, "Gender": args.gender,
        "Total_Bilirubin": args.total_bilirubin, "Direct_Bilirubin": args.direct_bilirubin,
        "Alkaline_Phosphotase": args.alkphos, "Alamine_Aminotransferase": args.alamine,
        "Aspartate_Aminotransferase": args.aspartate, "Total_Proteins": args.total_proteins,
        "Albumin": args.albumin, "Albumin_and_Globulin_Ratio": args.ag_ratio,
    }
    if all(v is not None for v in mapping.values()):
        return mapping
    return None


def collect_interactively():
    print("Enter patient values (press Ctrl+C to cancel):\n")
    values = {}
    for key, (label, cast) in PROMPTS.items():
        while True:
            raw = input(f"  {label}: ").strip()
            try:
                values[key] = raw if cast is str else cast(raw)
                break
            except ValueError:
                print("    Please enter a valid number.")
    return values


def main():
    args = parse_args()
    values = collect_from_args(args)
    if values is None:
        values = collect_interactively()

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"\nERROR: model/scaler not found at '{MODEL_PATH}' / '{SCALER_PATH}'.")
        print("Run 'python liver_disease_ensemble.py' first to train and save them.")
        sys.exit(1)

    gender_code = 1 if str(values["Gender"]).lower() == "male" else 0
    row = {**values, "Gender": gender_code}
    X_new = pd.DataFrame([row])[FEATURE_NAMES]
    X_scaled = pd.DataFrame(scaler.transform(X_new), columns=FEATURE_NAMES)

    proba = model.predict_proba(X_scaled)[0, 1]
    pred = int(proba >= 0.5)

    print("\n" + "=" * 50)
    print("PREDICTION  (model: Soft Voting Ensemble)")
    print("=" * 50)
    label = "LIVER DISEASE" if pred == 1 else "NO LIVER DISEASE"
    print(f"Result: {label}")
    print(f"Predicted probability of disease: {proba:.1%}")

    if not args.fast:
        try:
            background = joblib.load(BACKGROUND_PATH)
        except FileNotFoundError:
            background = None

        if background is not None:
            explainer = shap.KernelExplainer(model.predict_proba, background)
            sv = explainer.shap_values(X_scaled, nsamples=100, silent=True)
            sv = np.array(sv)
            # shape is (1, n_features, n_classes) -> take class 1 (disease), row 0
            if sv.ndim == 3:
                contrib = sv[0, :, 1]
            else:  # older SHAP versions return a list [class0_array, class1_array]
                contrib = sv[1][0] if isinstance(sv, list) else sv[0]

            contributions = pd.Series(contrib, index=FEATURE_NAMES).sort_values(key=np.abs, ascending=False)
            print("\nTop factors influencing this prediction (positive = pushes toward disease):")
            for feat, val in contributions.head(5).items():
                direction = "toward disease" if val > 0 else "toward no disease"
                print(f"  {feat:30s} {val:+.4f}  ({direction}, patient value = {values[feat]})")
        else:
            print("\n(SHAP background not found -- skipping explanation. "
                  "Run liver_disease_ensemble.py to generate it.)")

    print("\nNote: this is an educational/research model, not a validated diagnostic tool.")
    print("Consult a qualified clinician for actual medical decisions.")


if __name__ == "__main__":
    main()
