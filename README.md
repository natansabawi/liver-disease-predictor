# Liver Disease Predictor

An AI-powered Flask web app that estimates liver disease risk from routine liver
function test (LFT) values and returns clinical-style recommendations for the
reviewing doctor. Built on the Indian Liver Patient Dataset (ILPD).

> ⚠️ **Disclaimer:** This tool is for educational/screening-support purposes
> only. It is not a diagnostic device and must not replace clinical judgment,
> laboratory confirmation, or a licensed physician's evaluation.

## Features

- Predicts likely liver disease from 10 standard LFT inputs (bilirubin, ALT,
  AST, albumin, A/G ratio, etc.) using a trained `RandomForestClassifier`
- Confidence score and a HIGH / MODERATE / LOW risk severity badge
- Flags individual abnormal markers and generates targeted clinical
  recommendations (e.g. order HBsAg/HCV panel, AST/ALT ratio check, refer to
  hepatology)
- Responsive, animated single-page UI

## Project structure

```
.
├── app.py                          # Flask app + prediction/recommendation logic
├── main.py                         # Trains the RandomForest model from the dataset
├── predict.py                      # Simple CLI script for one-off predictions
├── model.pkl                       # Trained model (RandomForestClassifier)
├── scaler.pkl                      # Fitted StandardScaler
├── requirements.txt
├── data/
│   └── indian_liver_patient.csv    # Indian Liver Patient Dataset
├── templates/
│   └── index.html                  # UI template
└── static/
    ├── doctor.png                  # Header doctor photo
    └── logo.png                    # Org logo
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Retrain the model (optional)

The trained `model.pkl` / `scaler.pkl` are already included, but you can
retrain from scratch:

```bash
python main.py
```

This reads `data/indian_liver_patient.csv`, trains a `RandomForestClassifier`,
and overwrites `model.pkl` and `scaler.pkl`.

## Command-line prediction (optional)

`predict.py` runs a single prediction against the saved model without
starting the web server — edit the `patient` dict at the top of the file with
your own values, then run:

```bash
python predict.py
```

## Model

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Features:** Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline
  Phosphotase, ALT (SGPT), AST (SGOT), Total Proteins, Albumin, A/G Ratio
- **Dataset:** [Indian Liver Patient Dataset (ILPD)](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)

## Credits

Built by the **Ethiopian Giftedness and Talent Development Center**.
