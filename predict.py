import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# === ENTER PATIENT DATA HERE ===
patient = {
    'Age': [55],
    'Gender': [1],        # 1 = Male, 0 = Female
    'Total_Bilirubin': [2.5],
    'Direct_Bilirubin': [1.2],
    'Alkaline_Phosphotase': [250],
    'Alamine_Aminotransferase': [80],
    'Aspartate_Aminotransferase': [120],
    'Total_Protiens': [6.5],
    'Albumin': [3.2],
    'Albumin_and_Globulin_Ratio': [0.9]
}

# Predict
df = pd.DataFrame(patient)
df_scaled = scaler.transform(df)
result = model.predict(df_scaled)[0]

if result == 1:
    print("🔴 LIVER DISEASE")
else:
    print("🟢 NO DISEASE")