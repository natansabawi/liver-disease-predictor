# Liver Disease Classification using Machine Learning

Predicts whether a patient has liver disease from 10 clinical/blood-test attributes,
using the **Indian Liver Patient Dataset (ILPD)** from the UCI Machine Learning Repository
(583 patients: 416 with liver disease, 167 without).

## Files

```
liver_disease_project/
├── data/
│   └── indian_liver_patient.csv               # raw dataset (no header)
├── liver_disease_classification.py            # Stage 1: baseline pipeline (EDA + 8 models)
├── liver_disease_classification_advanced.py   # Stage 2: SMOTE + GridSearchCV tuning
├── liver_disease_interpretability.py          # Stage 3: SHAP explanations
├── liver_disease_ensemble.py                  # Stage 4: Voting & Stacking ensembles
├── predict_patient.py                         # CLI for scoring a new patient
├── Liver_Disease_Classification_Report.docx   # written summary report
├── tests/                                     # pytest suite (see "Testing" below)
├── pytest.ini
├── .coveragerc
├── requirements.txt
├── outputs/                    # Stage 1 outputs (plots, model_comparison.csv, saved model)
├── outputs_advanced/           # Stage 2 outputs (tuning comparison, saved tuned model)
├── outputs_interpretability/   # Stage 3 outputs (SHAP plots, RF model for SHAP)
└── outputs_ensemble/           # Stage 4 outputs (ensemble comparison, best model, scaler)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python liver_disease_classification.py
```

This will:
1. Load and clean the data (median-impute the few missing `Albumin_and_Globulin_Ratio` values, encode Gender).
2. Save EDA plots (class balance, correlation heatmap, age distribution).
3. Train/test split (80/20, stratified) with feature scaling.
4. Train 7–8 classifiers (Logistic Regression, Decision Tree, Random Forest,
   Gradient Boosting, SVM, KNN, Naive Bayes, XGBoost if installed) with 5-fold
   stratified cross-validation.
5. Evaluate every model on the held-out test set: accuracy, precision, recall,
   F1, ROC-AUC — saved to `outputs/model_comparison.csv`.
6. Plot confusion matrices and overlaid ROC curves for all models.
7. Plot feature importance (Random Forest).
8. Save the best-performing model (by test ROC-AUC) and the scaler with `joblib`,
   so you can reload them for inference without retraining.

## Results (example run, random_state=42)

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|
| SVM (RBF) | 0.69 | 0.74 | **0.84** |
| Logistic Regression | 0.74 | 0.79 | 0.83 |
| Naive Bayes | 0.64 | 0.67 | 0.80 |
| Random Forest | 0.74 | **0.83** | 0.78 |

SVM (RBF) had the best ROC-AUC in this run; Random Forest had the best F1.
Exact numbers vary slightly by random seed since the dataset is small (583 rows)
and imbalanced (~71% disease / 29% no-disease).

## Loading the saved model for new predictions

```python
import joblib
import pandas as pd

model = joblib.load("outputs/best_model_svm_(rbf).joblib")
scaler = joblib.load("outputs/scaler.joblib")

# new_patient must have the same columns/order as training features:
# Age, Gender(0=Female/1=Male), Total_Bilirubin, Direct_Bilirubin,
# Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase,
# Total_Proteins, Albumin, Albumin_and_Globulin_Ratio
new_patient = pd.DataFrame([{
    "Age": 45, "Gender": 1, "Total_Bilirubin": 1.2, "Direct_Bilirubin": 0.4,
    "Alkaline_Phosphotase": 200, "Alamine_Aminotransferase": 30,
    "Aspartate_Aminotransferase": 35, "Total_Proteins": 7.0,
    "Albumin": 3.5, "Albumin_and_Globulin_Ratio": 1.0,
}])
scaled = scaler.transform(new_patient)
prediction = model.predict(scaled)       # 1 = disease, 0 = no disease
probability = model.predict_proba(scaled)[:, 1]
```

## Advanced pipeline: SMOTE + hyperparameter tuning

`liver_disease_classification_advanced.py` extends the baseline with:

1. **SMOTE** oversampling of the minority class (no-disease) on the *training* set only —
   the test set is left untouched so evaluation reflects real-world class balance.
2. **GridSearchCV** (5-fold, scoring=`roc_auc`) tuning for the three strongest model
   families from the baseline run: Logistic Regression, Random Forest, SVM (RBF).
3. A **baseline vs. tuned+SMOTE comparison** (`outputs_advanced/baseline_vs_tuned_comparison.csv`
   and `.png`) so you can see exactly what tuning bought you, model by model.

Run it the same way:
```bash
python liver_disease_classification_advanced.py
```

Outputs go to `outputs_advanced/`: comparison table/chart, ROC curves, confusion matrix
for the best tuned model, a `best_hyperparameters.txt` reference file, and the saved
best tuned model + scaler.

### Example result (random_state=42)

| Model | Baseline ROC-AUC | Tuned+SMOTE ROC-AUC |
|---|---|---|
| Logistic Regression | 0.831 | 0.831 |
| Random Forest | 0.748 | 0.784 |
| SVM (RBF) | 0.671 | 0.798 |

Tuning + SMOTE gave a clear ROC-AUC boost for Random Forest and SVM, while Logistic
Regression (which was already well-calibrated with `class_weight="balanced"`) stayed flat.
F1 didn't improve uniformly — SMOTE shifts the decision boundary toward better minority-class
recall, which can trade off against precision, so the right choice depends on whether
missing a disease case (false negative) or a false alarm (false positive) is costlier
in your use case.

## Model interpretability (SHAP)

`liver_disease_interpretability.py` trains a Random Forest and explains its
predictions with SHAP (`TreeExplainer`, exact and fast for tree models):

- **Global importance** (`outputs_interpretability/shap_global_importance.png`) —
  which features matter most on average. Top drivers: Alkaline Phosphotase, Age,
  Aspartate Aminotransferase, Total Bilirubin, Alamine Aminotransferase — all
  clinically sensible (liver enzymes + bilirubin are the classic liver-function markers).
- **Beeswarm summary** (`shap_summary_beeswarm.png`) — shows *direction*, not just
  magnitude: high bilirubin/enzyme values push toward a disease prediction, as expected.
- **Per-patient waterfall plots** for three example cases (a confident correct
  disease prediction, a confident correct healthy prediction, and the model's most
  confident mistake) — showing exactly which lab values drove that individual prediction.

Run:
```bash
python liver_disease_interpretability.py
```

## Ensemble models

`liver_disease_ensemble.py` combines the three tuned base models (Logistic
Regression, Random Forest, SVM — using the hyperparameters found by
GridSearchCV) into two ensembles:

- **Soft Voting** — averages the base models' predicted probabilities.
- **Stacking** — trains a Logistic Regression meta-model on top of the base
  models' out-of-fold predictions.

Run:
```bash
python liver_disease_ensemble.py
```

### Result (random_state=42)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Voting Ensemble** | **0.786** | **0.847** | 0.831 |
| Logistic Regression | 0.718 | 0.769 | 0.831 |
| Stacking Ensemble | 0.744 | 0.833 | 0.799 |
| SVM (RBF) | 0.726 | 0.787 | 0.798 |
| Random Forest | 0.744 | 0.828 | 0.784 |

The **Voting Ensemble is the new best overall model** in this project — it
matches Logistic Regression's ROC-AUC while beating every individual model on
accuracy and F1, with much more balanced precision/recall on the minority
("no disease") class (0.62 precision / 0.68 recall vs. Logistic Regression's
0.92 precision / 0.66 recall alone). Averaging probabilities across models
with different error patterns (linear vs. tree-based vs. kernel) smooths out
each one's individual weaknesses.

Outputs: `outputs_ensemble/ensemble_comparison.csv`, ROC curve overlay,
confusion matrix, bar chart, and the saved best model + scaler.

## Predicting a new patient (CLI)

`predict_patient.py` loads the project's best overall model — the **Soft
Voting Ensemble** — and predicts risk for one patient, with a SHAP
explanation of the top 5 contributing factors. Run `liver_disease_ensemble.py`
first (it saves the model, scaler, and a SHAP background sample this script needs).

Since the Voting Ensemble mixes model types (linear, tree, kernel), it can't
use SHAP's fast `TreeExplainer` — this script uses `KernelExplainer` against a
small k-means-summarized background instead, which stays fast (well under a
second) for one-off, single-patient explanations.

Interactive:
```bash
python predict_patient.py
```

Scripted (all flags):
```bash
python predict_patient.py --age 60 --gender Male --total_bilirubin 8.5 \
    --direct_bilirubin 4.2 --alkphos 450 --alamine 90 --aspartate 110 \
    --total_proteins 6.0 --albumin 2.8 --ag_ratio 0.6
```

Add `--fast` to skip the SHAP explanation and just get the prediction:
```bash
python predict_patient.py --fast --age 60 --gender Male --total_bilirubin 8.5 \
    --direct_bilirubin 4.2 --alkphos 450 --alamine 90 --aspartate 110 \
    --total_proteins 6.0 --albumin 2.8 --ag_ratio 0.6
```

## Testing

The `tests/` directory has a pytest suite covering data preprocessing,
each pipeline stage (baseline, SMOTE+tuning, ensembling), the CLI's argument
handling, and integration checks against the saved model artifacts. Unit
tests run against a small synthetic dataset (fast, no dependency on trained
models); artifact tests skip cleanly if `liver_disease_ensemble.py` hasn't
been run yet, rather than failing the suite.

Run all tests:
```bash
python -m pytest
```

Run with a coverage report:
```bash
python -m pytest --cov=. --cov-report=term-missing
```

## Notes & limitations

- The dataset is small (583 rows) and imbalanced (~71% disease / 29% no-disease), so
  metrics vary noticeably with the random seed — treat any single run's numbers as
  indicative, not definitive.
- This is a research/educational pipeline, not a validated clinical diagnostic tool.
