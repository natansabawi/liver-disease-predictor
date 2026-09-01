from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def get_recommendations(patient, prediction):
    recommendations = []
    risk_factors = []
    severity = "LOW"

    if patient['Total_Bilirubin'][0] > 2.0:
        risk_factors.append("High Total Bilirubin")
        recommendations.append("Order abdominal ultrasound to assess liver structure and bile ducts")
        severity = "HIGH"
    elif patient['Total_Bilirubin'][0] > 1.2:
        risk_factors.append("Elevated Total Bilirubin")
        recommendations.append("Monitor bilirubin levels weekly")
        if severity == "LOW": severity = "MODERATE"

    if patient['Direct_Bilirubin'][0] > 0.5:
        risk_factors.append("High Direct Bilirubin")
        recommendations.append("Evaluate for cholestasis or biliary obstruction")
        severity = "HIGH"

    if patient['Alamine_Aminotransferase'][0] > 100:
        risk_factors.append("Very High ALT (SGPT)")
        recommendations.append("Order Hepatitis B surface antigen (HBsAg) and Hepatitis C antibody")
        recommendations.append("Consider liver function panel recheck in 1 week")
        severity = "HIGH"
    elif patient['Alamine_Aminotransferase'][0] > 56:
        risk_factors.append("Elevated ALT (SGPT)")
        recommendations.append("Assess for fatty liver disease (NAFLD) or medication-induced injury")
        if severity == "LOW": severity = "MODERATE"

    if patient['Aspartate_Aminotransferase'][0] > 100:
        risk_factors.append("Very High AST (SGOT)")
        recommendations.append("Check AST/ALT ratio; if >2, suspect alcoholic liver disease")
        recommendations.append("Order coagulation profile (PT/INR) to assess liver synthetic function")
        severity = "HIGH"
    elif patient['Aspartate_Aminotransferase'][0] > 40:
        risk_factors.append("Elevated AST (SGOT)")
        if severity == "LOW": severity = "MODERATE"

    if patient['Alkaline_Phosphotase'][0] > 200:
        risk_factors.append("High Alkaline Phosphatase")
        recommendations.append("Order GGT and 5'-nucleotidase to confirm hepatic origin")
        recommendations.append("Evaluate for primary biliary cholangitis or biliary obstruction")
        severity = "HIGH"
    elif patient['Alkaline_Phosphotase'][0] > 147:
        risk_factors.append("Elevated Alkaline Phosphatase")
        if severity == "LOW": severity = "MODERATE"

    if patient['Albumin'][0] < 3.0:
        risk_factors.append("Low Albumin")
        recommendations.append("Assess nutritional status and protein intake")
        recommendations.append("Check for chronic liver disease or nephrotic syndrome")
        severity = "HIGH"
    elif patient['Albumin'][0] < 3.5:
        risk_factors.append("Low-Normal Albumin")
        if severity == "LOW": severity = "MODERATE"

    if patient['Albumin_and_Globulin_Ratio'][0] < 0.8:
        risk_factors.append("Low A/G Ratio")
        recommendations.append("Suggest serum protein electrophoresis to rule out multiple myeloma")
        recommendations.append("Evaluate for chronic liver disease or immune disorders")
        severity = "HIGH"
    elif patient['Albumin_and_Globulin_Ratio'][0] < 1.0:
        risk_factors.append("Reduced A/G Ratio")
        if severity == "LOW": severity = "MODERATE"

    if patient['Total_Protiens'][0] < 5.5:
        risk_factors.append("Low Total Proteins")
        recommendations.append("Evaluate for malnutrition, chronic infection, or liver/kidney disease")
        severity = "HIGH"

    if prediction == 1:
        if severity == "LOW": severity = "MODERATE"
        recommendations.insert(0, "LIVER DISEASE PREDICTED - Immediate clinical correlation required")
        recommendations.append("Refer to hepatology/gastroenterology for further evaluation")
        recommendations.append("Consider FibroScan or liver biopsy if clinically indicated")
        recommendations.append("Advise absolute alcohol abstinence")
        recommendations.append("Screen for diabetes and metabolic syndrome (common comorbidities)")
    else:
        if len(risk_factors) == 0:
            recommendations.append("All liver function parameters within acceptable range")
            recommendations.append("Continue routine annual health screening")
            recommendations.append("Maintain healthy lifestyle: balanced diet, regular exercise, limit alcohol")
        else:
            recommendations.insert(0, "No liver disease predicted, but some markers are abnormal")
            recommendations.append("Recommend lifestyle modification and repeat LFT in 4-6 weeks")
            recommendations.append("Review current medications for hepatotoxicity potential")

    return {'severity': severity, 'risk_factors': risk_factors, 'recommendations': recommendations}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    patient = {
        'Age': [float(request.form['age'])],
        'Gender': [1 if request.form['gender'] == 'Male' else 0],
        'Total_Bilirubin': [float(request.form['tb'])],
        'Direct_Bilirubin': [float(request.form['db'])],
        'Alkaline_Phosphotase': [float(request.form['alp'])],
        'Alamine_Aminotransferase': [float(request.form['alt'])],
        'Aspartate_Aminotransferase': [float(request.form['ast'])],
        'Total_Protiens': [float(request.form['tp'])],
        'Albumin': [float(request.form['albumin'])],
        'Albumin_and_Globulin_Ratio': [float(request.form['ag_ratio'])]
    }

    df = pd.DataFrame(patient)
    df_scaled = scaler.transform(df)
    result = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0]

    confidence = prob[0] if result == 1 else prob[1]

    if result == 1:
        output = "LIVER DISEASE DETECTED"
        css_class = "disease"
        icon = "WARNING"
    else:
        output = "NO LIVER DISEASE"
        css_class = "healthy"
        icon = "OK"

    rec_data = get_recommendations(patient, result)

    return render_template('index.html', 
                           prediction=output, 
                           confidence=round(confidence*100, 2),
                           css_class=css_class,
                           icon=icon,
                           severity=rec_data['severity'],
                           risk_factors=rec_data['risk_factors'],
                           recommendations=rec_data['recommendations'])

if __name__ == '__main__':
    app.run(debug=True)
