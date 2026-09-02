import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="🩺",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(APP_DIR / "model.pkl")
    scaler = joblib.load(APP_DIR / "scaler.pkl")
    return model, scaler


@st.cache_data
def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Clinical recommendation logic (ported from the Flask app)
# ---------------------------------------------------------------------------

def get_recommendations(patient, prediction):
    recommendations = []
    risk_factors = []
    severity = "LOW"

    if patient["Total_Bilirubin"] > 2.0:
        risk_factors.append("High Total Bilirubin")
        recommendations.append("Order abdominal ultrasound to assess liver structure and bile ducts")
        severity = "HIGH"
    elif patient["Total_Bilirubin"] > 1.2:
        risk_factors.append("Elevated Total Bilirubin")
        recommendations.append("Monitor bilirubin levels weekly")
        if severity == "LOW":
            severity = "MODERATE"

    if patient["Direct_Bilirubin"] > 0.5:
        risk_factors.append("High Direct Bilirubin")
        recommendations.append("Evaluate for cholestasis or biliary obstruction")
        severity = "HIGH"

    if patient["Alamine_Aminotransferase"] > 100:
        risk_factors.append("Very High ALT (SGPT)")
        recommendations.append("Order Hepatitis B surface antigen (HBsAg) and Hepatitis C antibody")
        recommendations.append("Consider liver function panel recheck in 1 week")
        severity = "HIGH"
    elif patient["Alamine_Aminotransferase"] > 56:
        risk_factors.append("Elevated ALT (SGPT)")
        recommendations.append("Assess for fatty liver disease (NAFLD) or medication-induced injury")
        if severity == "LOW":
            severity = "MODERATE"

    if patient["Aspartate_Aminotransferase"] > 100:
        risk_factors.append("Very High AST (SGOT)")
        recommendations.append("Check AST/ALT ratio; if >2, suspect alcoholic liver disease")
        recommendations.append("Order coagulation profile (PT/INR) to assess liver synthetic function")
        severity = "HIGH"
    elif patient["Aspartate_Aminotransferase"] > 40:
        risk_factors.append("Elevated AST (SGOT)")
        if severity == "LOW":
            severity = "MODERATE"

    if patient["Alkaline_Phosphotase"] > 200:
        risk_factors.append("High Alkaline Phosphatase")
        recommendations.append("Order GGT and 5'-nucleotidase to confirm hepatic origin")
        recommendations.append("Evaluate for primary biliary cholangitis or biliary obstruction")
        severity = "HIGH"
    elif patient["Alkaline_Phosphotase"] > 147:
        risk_factors.append("Elevated Alkaline Phosphatase")
        if severity == "LOW":
            severity = "MODERATE"

    if patient["Albumin"] < 3.0:
        risk_factors.append("Low Albumin")
        recommendations.append("Assess nutritional status and protein intake")
        recommendations.append("Check for chronic liver disease or nephrotic syndrome")
        severity = "HIGH"
    elif patient["Albumin"] < 3.5:
        risk_factors.append("Low-Normal Albumin")
        if severity == "LOW":
            severity = "MODERATE"

    if patient["Albumin_and_Globulin_Ratio"] < 0.8:
        risk_factors.append("Low A/G Ratio")
        recommendations.append("Suggest serum protein electrophoresis to rule out multiple myeloma")
        recommendations.append("Evaluate for chronic liver disease or immune disorders")
        severity = "HIGH"
    elif patient["Albumin_and_Globulin_Ratio"] < 1.0:
        risk_factors.append("Reduced A/G Ratio")
        if severity == "LOW":
            severity = "MODERATE"

    if patient["Total_Protiens"] < 5.5:
        risk_factors.append("Low Total Proteins")
        recommendations.append("Evaluate for malnutrition, chronic infection, or liver/kidney disease")
        severity = "HIGH"

    if prediction == 1:
        if severity == "LOW":
            severity = "MODERATE"
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

    return {"severity": severity, "risk_factors": risk_factors, "recommendations": recommendations}


# ---------------------------------------------------------------------------
# Global styling (gradient theme, button, card & badge styles)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        padding: 2rem 2.5rem 3rem 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }

    div.stButton > button {
        width: 100%;
        padding: 0.9rem;
        margin-top: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        color: white;
    }

    .result-card {
        margin-top: 1.5rem;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
    }
    .result-card.disease {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    .result-card.healthy {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
    }
    .result-icon { font-size: 50px; margin-bottom: 10px; }
    .result-title { font-size: 26px; font-weight: 700; }
    .result-confidence { font-size: 17px; opacity: 0.9; margin-bottom: 10px; }

    .severity-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 10px;
        border: 3px solid white;
    }
    .severity-HIGH { background: #8b0000; }
    .severity-MODERATE { background: #ff8c00; }
    .severity-LOW { background: #006400; }

    .recommendations-panel {
        margin-top: 1.5rem;
        padding: 30px;
        border-radius: 20px;
        background: #f8f9ff;
        border: 2px solid #e0e5ff;
    }
    .recommendations-panel h3 { color: #333; font-size: 20px; margin-bottom: 20px; }

    .risk-factors { margin-bottom: 20px; }
    .risk-factors h4 { color: #e74c3c; font-size: 14px; text-transform: uppercase; margin-bottom: 10px; }
    .risk-tag {
        display: inline-block;
        background: #ffe0e0;
        color: #c0392b;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px;
        border: 1px solid #ffcdd2;
    }

    .recommendations-list { list-style: none; padding: 0; margin: 0; }
    .recommendations-list li {
        padding: 14px 18px;
        margin-bottom: 10px;
        background: white;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        font-size: 14px;
        color: #444;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .recommendations-list li:first-child {
        border-left-color: #e74c3c;
        background: #fff5f5;
        font-weight: 600;
        color: #c0392b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Animated header (doctor photo + pulsing liver badge + orbiting magnifier)
# rendered in an isolated component so its keyframe animations don't clash
# with Streamlit's own styles.
# ---------------------------------------------------------------------------

doctor_path = APP_DIR / "static" / "doctor.png"
logo_path = APP_DIR / "static" / "logo.png"
doctor_b64 = image_to_base64(doctor_path) if doctor_path.exists() else ""
logo_b64 = image_to_base64(logo_path) if logo_path.exists() else ""

HEADER_HTML = f"""
<html>
<head>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Poppins', sans-serif; }}

    .header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        text-align: center;
        border-radius: 24px;
        position: relative;
        overflow: hidden;
    }}
    .header-content {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        position: relative;
        z-index: 2;
    }}

    .doctor-container {{ position: relative; width: 180px; height: 180px; flex-shrink: 0; }}
    .doctor-photo {{
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        display: block;
        border: 5px solid rgba(255,255,255,0.9);
        box-shadow: 0 12px 30px rgba(0,0,0,0.3);
        box-sizing: border-box;
    }}

    .liver-badge-ring {{
        position: absolute; right: -12px; bottom: -10px;
        width: 62px; height: 62px; border-radius: 50%;
        background: #ffffff; box-shadow: 0 6px 16px rgba(0,0,0,0.25); z-index: 2;
    }}
    .liver-wrap {{
        position: absolute; right: -2px; bottom: 3px;
        width: 44px; height: 36px; z-index: 3;
        background: linear-gradient(135deg, #c96a3e 0%, #a24a26 55%, #7a3517 100%);
        border-radius: 60% 40% 55% 45% / 55% 45% 60% 40%;
        box-shadow: 0 5px 14px rgba(0,0,0,0.15);
        animation: liverPulse 2.2s ease-in-out infinite;
    }}
    .liver-wrap::before {{
        content: ''; position: absolute; top: 10px; left: -12px;
        width: 22px; height: 18px;
        background: linear-gradient(135deg, #c96a3e 0%, #8b4513 100%);
        border-radius: 60% 40% 50% 50% / 60% 40% 60% 40%;
    }}
    .liver-wrap::after {{
        content: ''; position: absolute; top: 6px; left: 10px;
        width: 15px; height: 8px; background: rgba(255,255,255,0.35);
        border-radius: 50%; filter: blur(1px);
    }}
    @keyframes liverPulse {{
        0%, 100% {{ transform: scale(1); box-shadow: 0 5px 14px rgba(0,0,0,0.15), 0 0 0 rgba(255,140,60,0); }}
        50% {{ transform: scale(1.1); box-shadow: 0 5px 18px rgba(0,0,0,0.2), 0 0 18px rgba(255,140,60,0.65); }}
    }}

    .magnifier-orbit {{
        position: absolute; right: -18px; bottom: -16px;
        width: 76px; height: 76px; z-index: 4; pointer-events: none;
        animation: orbitRotate 4.5s linear infinite;
    }}
    .magnifier {{
        position: absolute; top: 0; left: 50%; margin-left: -10px;
        width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.9);
        border-radius: 50%; background: rgba(255,255,255,0.15);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
        animation: counterRotate 4.5s linear infinite;
    }}
    .magnifier::after {{
        content: ''; position: absolute; bottom: -10px; right: -6px;
        width: 3px; height: 13px; background: rgba(255,255,255,0.9);
        border-radius: 2px; transform: rotate(45deg);
    }}
    @keyframes orbitRotate {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    @keyframes counterRotate {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(-360deg); }} }}

    .ekg-line {{
        position: absolute; bottom: 6px; left: 0; width: 100%; height: 30px;
        opacity: 0.4; z-index: 1; pointer-events: none;
    }}
    .ekg-line polyline {{
        fill: none; stroke: #ffffff; stroke-width: 2;
        stroke-linecap: round; stroke-linejoin: round;
        stroke-dasharray: 18 600; animation: ekgSweep 3.2s linear infinite;
    }}
    @keyframes ekgSweep {{ to {{ stroke-dashoffset: -618; }} }}

    .header-text {{ color: white; text-align: left; }}
    .header-text h1 {{ font-size: 30px; font-weight: 700; margin-bottom: 8px; }}
    .header-text p {{ color: rgba(255,255,255,0.9); font-size: 14px; line-height: 1.5; }}

    .logo {{
        width: 90px; height: 90px; border-radius: 50%;
        border: 4px solid white; box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} }}
</style>
</head>
<body>
    <div class="header">
        <svg class="ekg-line" viewBox="0 0 300 40" preserveAspectRatio="none">
            <polyline points="0,20 40,20 55,4 68,36 80,20 110,20 122,8 135,32 148,20 300,20"></polyline>
        </svg>
        <div class="header-content">
            <div class="doctor-container">
                <img class="doctor-photo" src="data:image/png;base64,{doctor_b64}" alt="Doctor">
                <div class="liver-badge-ring"></div>
                <div class="liver-wrap"></div>
                <div class="magnifier-orbit"><div class="magnifier"></div></div>
            </div>
            <div class="header-text">
                <h1>Liver Disease Predictor</h1>
                <p>AI-Powered Healthcare Assistant</p>
                <p>Ethiopian Giftedness &amp; Talent Development Center</p>
            </div>
            <img class="logo" src="data:image/png;base64,{logo_b64}" alt="Logo">
        </div>
    </div>
</body>
</html>
"""

components.html(HEADER_HTML, height=300, scrolling=False)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

st.markdown("### Patient Liver Function Panel")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=55)
        tb = st.number_input("Total Bilirubin", min_value=0.0, value=2.5, step=0.1, format="%.1f")
        alp = st.number_input("Alkaline Phosphotase", min_value=0, value=250)
        ast = st.number_input("AST (SGOT)", min_value=0, value=120)
        albumin = st.number_input("Albumin", min_value=0.0, value=3.2, step=0.1, format="%.1f")
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        db = st.number_input("Direct Bilirubin", min_value=0.0, value=1.2, step=0.1, format="%.1f")
        alt = st.number_input("ALT (SGPT)", min_value=0, value=80)
        tp = st.number_input("Total Proteins", min_value=0.0, value=6.5, step=0.1, format="%.1f")
        ag_ratio = st.number_input("A/G Ratio", min_value=0.0, value=0.9, step=0.01, format="%.2f")

    submitted = st.form_submit_button("ANALYZE PATIENT")

# ---------------------------------------------------------------------------
# Prediction + results
# ---------------------------------------------------------------------------

if submitted:
    model, scaler = load_model_and_scaler()

    patient = {
        "Age": float(age),
        "Gender": 1 if gender == "Male" else 0,
        "Total_Bilirubin": float(tb),
        "Direct_Bilirubin": float(db),
        "Alkaline_Phosphotase": float(alp),
        "Alamine_Aminotransferase": float(alt),
        "Aspartate_Aminotransferase": float(ast),
        "Total_Protiens": float(tp),
        "Albumin": float(albumin),
        "Albumin_and_Globulin_Ratio": float(ag_ratio),
    }

    df = pd.DataFrame([patient])
    df_scaled = scaler.transform(df)
    result = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0]
    confidence = prob[0] if result == 1 else prob[1]

    if result == 1:
        output, css_class, icon = "LIVER DISEASE DETECTED", "disease", "⚠️"
    else:
        output, css_class, icon = "NO LIVER DISEASE", "healthy", "✅"

    rec_data = get_recommendations(patient, result)

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-title">{output}</div>
            <div class="result-confidence">Confidence: {round(confidence * 100, 2)}%</div>
            <div class="severity-badge severity-{rec_data['severity']}">{rec_data['severity']} RISK</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_tags_html = "".join(f'<span class="risk-tag">{f}</span>' for f in rec_data["risk_factors"])
    risk_section_html = (
        f'<div class="risk-factors"><h4>Abnormal Markers Detected:</h4>{risk_tags_html}</div>'
        if rec_data["risk_factors"]
        else ""
    )
    recs_html = "".join(f"<li>{r}</li>" for r in rec_data["recommendations"])

    st.markdown(
        f"""
        <div class="recommendations-panel">
            <h3>Clinical Recommendations for Doctor</h3>
            {risk_section_html}
            <ul class="recommendations-list">{recs_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<p style='text-align:center; color:#999; font-size:12px; margin-top:2rem;'>"
    "© 2026 Ethiopian Giftedness &amp; Talent Development Center | AI-Powered Healthcare</p>",
    unsafe_allow_html=True,
)
