import pandas as pd
import numpy as np
import re, os, datetime, sqlite3, requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import importlib.util
from fpdf import FPDF
import shap
from xhtml2pdf import pisa

DB_PATH = "escalations.db"

# 🔮 ETA Prediction
def predict_resolution_eta(df):
    df = df.copy()
    df = df[df['status'] == 'Resolved']
    df['duration_hours'] = (pd.to_datetime(df['status_update_date']) - pd.to_datetime(df['timestamp'])).dt.total_seconds() / 3600
    features = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    model = LinearRegression()
    model.fit(features, df['duration_hours'])

    def predict(case):
        X = pd.get_dummies(pd.DataFrame([case]))
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        return round(model.predict(X)[0], 2)

    return predict

# 🧠 SHAP Explanation
def show_shap_explanation(model, case_features):
    explainer = shap.TreeExplainer(model)
    X = pd.get_dummies(pd.DataFrame([case_features]))
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    shap_values = explainer.shap_values(X)
    st.subheader("🔍 Likely to Escalate Explanation")
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], X, matplotlib=True)

# 🆕 SHAP Summary Plot
def generate_shap_plot(model=None, X_sample=None):
    try:
        if model is None or X_sample is None or X_sample.empty:
            st.info("No SHAP plot generated — missing model or sample data.")
            return
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown("### 🔎 SHAP Feature Impact on Likely to Escalate")
        st.pyplot(shap.summary_plot(shap_values, X_sample))
    except Exception as e:
        st.error(f"SHAP plot generation failed: {e}")

# 📄 PDF Report Generator
def generate_pdf_report():
    df = fetch_escalations()
    html = f"""
    <html>
    <head>
    <style>
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: 8px;
        text-align: left;
    }}
    th {{
        background-color: #f2f2f2;
    }}
    h2 {{
        text-align: center;
        color: #2c3e50;
    }}
    </style>
    </head>
    <body>
    <h2>📄 Likely to Escalate Report</h2>
    {df.to_html(index=False)}
    </body>
    </html>
    """
    try:
        with open("report.pdf", "wb") as f:
            pisa.CreatePDF(html, dest=f)
        st.success("✅ PDF report generated successfully.")
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")

# 📊 Text Summary PDF
def generate_text_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Likely to Escalate Summary", ln=True)
    pdf.cell(200, 10, txt=f"Generated: {datetime.datetime.now()}", ln=True)
    for _, row in df.iterrows():
        pdf.multi_cell(0, 10, txt=f"{row['id']}: {row['issue']}")
    pdf.output("summary_report.pdf")

# 📊 Model Metrics
def render_model_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.markdown("### 📊 Model Performance on Likely to Escalate Prediction")
    st.json(report)

# 📝 Feedback Scoring
def score_feedback_quality(notes):
    if not notes:
        return 0
    score = len(notes.split()) / 10
    return min(score, 1.0)

# 🧪 Schema Validator
def validate_escalation_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    required_columns = ["owner_email", "status_update_date", "user_feedback", "likely_to_escalate"]
    for col in required_columns:
        try:
            cursor.execute(f"SELECT {col} FROM escalations LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute(f"ALTER TABLE escalations ADD COLUMN {col} TEXT")
    conn.commit()
    conn.close()

# 🧾 Audit Logger
def log_escalation_action(action_type, case_id, user, details):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        timestamp TEXT,
        action_type TEXT,
        case_id TEXT,
        user TEXT,
        details TEXT
    )
    ''')
    cursor.execute('''
    INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)
    ''', (datetime.datetime.now().isoformat(), action_type, case_id, user, details))
    conn.commit()
    conn.close()

# 🧬 Duplicate Detection
def detect_cosine_duplicates(df, threshold=0.85):
    issues = df['issue'].fillna("").tolist()
    vectorizer = TfidfVectorizer().fit_transform(issues)
    sim_matrix = cosine_similarity(vectorizer)
    duplicates = []
    for i in range(len(issues)):
        for j in range(i+1, len(issues)):
            if sim_matrix[i][j] > threshold:
                duplicates.append((df.iloc[i]['id'], df.iloc[j]['id']))
    return duplicates

# 📧 Email Thread Linking
def link_email_threads(df):
    df['thread_id'] = df['issue'].apply(lambda x: re.sub(r'Re:|Fwd:', '', x.split('-')[0]).strip().lower())
    return df.groupby('thread_id')

# 🔌 Plugin Loader
def load_custom_plugins(path="plugins/"):
    for file in os.listdir(path):
        if file.endswith(".py"):
            spec = importlib.util.spec_from_file_location(file[:-3], os.path.join(path, file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

# 📲 WhatsApp API
def send_whatsapp_message(phone, message):
    url = "https://api.twilio.com/..."  # Replace with actual API
    payload = {"to": phone, "body": message}
    headers = {"Authorization": "Bearer YOUR_TOKEN"}
    response = requests.post(url, json=payload, headers=headers)
    return response.status_code == 200

# 📥 DB Fetch Helper
def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df
