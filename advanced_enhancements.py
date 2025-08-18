# advanced_enhancements.py
# Utilities for analytics, SHAP, PDF generation, duplicate detection, and a robust audit log.

import os
import re
import datetime
import sqlite3
import requests
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import importlib.util

# Optional deps (guarded)
try:
    import shap
except Exception:
    shap = None

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

try:
    from xhtml2pdf import pisa
except Exception:
    pisa = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("ESCALATEAI_DB_PATH", "escalations.db")

def _get_conn():
    return sqlite3.connect(DB_PATH)

# -----------------------------------------------------------------------------
# ETA Prediction (simple baseline)
# -----------------------------------------------------------------------------
def predict_resolution_eta(df: pd.DataFrame):
    if df is None or df.empty:
        return lambda _case: np.nan

    data = df.copy()
    if "status" not in data.columns:
        return lambda _case: np.nan

    data = data[data["status"] == "Resolved"].copy()
    if data.empty:
        return lambda _case: np.nan

    for col in ("timestamp", "status_update_date"):
        if col not in data.columns:
            data[col] = pd.NaT

    data["duration_hours"] = (
        pd.to_datetime(data["status_update_date"], errors="coerce")
        - pd.to_datetime(data["timestamp"], errors="coerce")
    ).dt.total_seconds() / 3600.0
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["duration_hours"])
    if data.empty:
        return lambda _case: np.nan

    feature_cols = ["sentiment", "urgency", "severity", "criticality"]
    for c in feature_cols:
        if c not in data.columns:
            data[c] = "unknown"

    X = pd.get_dummies(data[feature_cols], dummy_na=True)
    y = data["duration_hours"].astype(float)
    if X.empty or y.empty:
        return lambda _case: np.nan

    model = LinearRegression()
    model.fit(X, y)

    def predict(case: dict):
        X1 = pd.get_dummies(pd.DataFrame([case], columns=feature_cols), dummy_na=True)
        X1 = X1.reindex(columns=model.feature_names_in_, fill_value=0)
        try:
            return float(np.round(model.predict(X1)[0], 2))
        except Exception:
            return np.nan

    return predict

# -----------------------------------------------------------------------------
# SHAP Explanation
# -----------------------------------------------------------------------------
def show_shap_explanation(model, case_features: dict):
    if shap is None:
        st.info("SHAP not installed; skipping explanation.")
        return
    try:
        explainer = shap.TreeExplainer(model)
        X = pd.get_dummies(pd.DataFrame([case_features]))
        X = X.reindex(columns=getattr(model, "feature_names_in_", X.columns), fill_value=0)
        shap_values = explainer.shap_values(X)

        st.subheader("🔍 Likely to Escalate — SHAP Explanation")
        try:
            shap.initjs()
        except Exception:
            pass

        if isinstance(shap_values, list) and len(shap_values) > 1:
            base_value = getattr(explainer, "expected_value", [0, 0])[1]
            sv = shap_values[1]
        else:
            base_value = getattr(explainer, "expected_value", 0)
            sv = shap_values

        shap.force_plot(base_value, sv, X, matplotlib=True)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

def generate_shap_plot(model=None, X_sample: pd.DataFrame = None):
    if shap is None:
        st.info("SHAP not installed; cannot generate plot.")
        return
    try:
        if model is None or X_sample is None or X_sample.empty:
            st.info("No SHAP plot generated — missing model or data sample.")
            return
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.markdown("### 🔎 SHAP Feature Impact on Likely to Escalate")
        shap.summary_plot(shap_values if isinstance(shap_values, list) else [shap_values],
                          X_sample, show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.error(f"SHAP plot generation failed: {e}")

# -----------------------------------------------------------------------------
# PDF Generators
# -----------------------------------------------------------------------------
def fetch_escalations() -> pd.DataFrame:
    conn = _get_conn()
    try:
        return pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def generate_pdf_report(output_path: str = "report.pdf"):
    if pisa is None:
        st.error("xhtml2pdf (pisa) not installed; cannot generate PDF.")
        return
    df = fetch_escalations()
    if df is None or df.empty:
        st.info("No data to include in the report.")
        return

    html = f"""
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
    body {{ font-family: Arial, sans-serif; font-size: 12px; color: #222; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; word-break: break-word; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; font-size: 11px; }}
    th {{ background-color: #f2f2f2; }}
    h2 {{ text-align: center; color: #2c3e50; }}
    </style>
    </head>
    <body>
    <h2>📄 Likely to Escalate Report</h2>
    {df.to_html(index=False, escape=False)}
    </body>
    </html>
    """
    try:
        with open(output_path, "wb") as f:
            pisa.CreatePDF(html, dest=f)
        st.success(f"✅ PDF report generated: {output_path}")
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")

def generate_text_pdf(df: pd.DataFrame, output_path: str = "summary_report.pdf"):
    if FPDF is None:
        st.error("fpdf not installed; cannot generate text PDF.")
        return
    if df is None or df.empty:
        st.info("No data to export.")
        return
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Likely to Escalate — Summary", ln=True)
        pdf.cell(0, 10, txt=f"Generated: {datetime.datetime.now().isoformat()}", ln=True)
        for _, row in df.iterrows():
            issue_txt = str(row.get("issue", "")).strip()
            rid = str(row.get("id", ""))
            pdf.multi_cell(0, 7, txt=f"{rid}: {issue_txt}")
        pdf.output(output_path)
        st.success(f"✅ PDF summary generated: {output_path}")
    except Exception as e:
        st.error(f"❌ PDF summary failed: {e}")

# -----------------------------------------------------------------------------
# Model Metrics
# -----------------------------------------------------------------------------
def render_model_metrics(model, X_test: pd.DataFrame, y_test: pd.Series):
    if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
        st.info("No test data available for metrics.")
        return
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.markdown("### 📊 Model Performance on Likely to Escalate Prediction")
        st.json(report)
    except Exception as e:
        st.error(f"Metrics rendering failed: {e}")

# -----------------------------------------------------------------------------
# Feedback Scoring
# -----------------------------------------------------------------------------
def score_feedback_quality(notes: str) -> float:
    if not notes:
        return 0.0
    score = len(str(notes).split()) / 10.0
    return float(min(score, 1.0))

# -----------------------------------------------------------------------------
# Escalations table schema validator/additions
# -----------------------------------------------------------------------------
def validate_escalation_schema():
    required_columns = ["owner_email", "status_update_date", "user_feedback", "likely_to_escalate"]
    conn = _get_conn()
    try:
        cur = conn.cursor()
        for col in required_columns:
            try:
                cur.execute(f"SELECT {col} FROM escalations LIMIT 1")
            except sqlite3.OperationalError:
                cur.execute(f"ALTER TABLE escalations ADD COLUMN {col} TEXT")
        conn.commit()
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Robust Audit Logger (schema-safe + auto-migration)
# -----------------------------------------------------------------------------
_AUDIT_TABLE = "audit_log"
_AUDIT_CANON_COLS = ["timestamp", "action_type", "case_id", "user", "details"]  # all TEXT

def _audit_table_info(cur) -> list:
    cur.execute(f"PRAGMA table_info({_AUDIT_TABLE})")
    return [(r[1], r[2]) for r in cur.fetchall()]

def _audit_create(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {_AUDIT_TABLE} (
            timestamp   TEXT NOT NULL,
            action_type TEXT NOT NULL,
            case_id     TEXT,
            user        TEXT,
            details     TEXT
        )
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{_AUDIT_TABLE}_ts ON {_AUDIT_TABLE}(timestamp)")

def _audit_migrate_if_needed(cur):
    info = _audit_table_info(cur)
    if not info:
        _audit_create(cur)
        return

    existing_cols = [c for c, _t in info]

    # If table already matches canonical 5 TEXT columns, just ensure index and return
    if all(c in existing_cols for c in _AUDIT_CANON_COLS) and len(existing_cols) == len(_AUDIT_CANON_COLS):
        _audit_create(cur)  # ensures index
        return

    # Otherwise rename & recreate to canonical schema, then copy intersecting cols
    cur.execute(f"ALTER TABLE {_AUDIT_TABLE} RENAME TO {_AUDIT_TABLE}_old")
    _audit_create(cur)

    intersection = [c for c in _AUDIT_CANON_COLS if c in existing_cols]
    if intersection:
        cols_csv = ", ".join(intersection)
        cur.execute(f"""
            INSERT INTO {_AUDIT_TABLE} ({cols_csv})
            SELECT {cols_csv} FROM {_AUDIT_TABLE}_old
        """)
    cur.execute(f"DROP TABLE IF EXISTS {_AUDIT_TABLE}_old")

def ensure_audit_log_table():
    conn = _get_conn()
    try:
        cur = conn.cursor()
        _audit_create(cur)
        _audit_migrate_if_needed(cur)
        conn.commit()
    finally:
        conn.close()

def log_escalation_action(action_type: str, case_id: str, user: str, details: str):
    """
    Append a row to the audit log using explicit columns (order-safe, type-safe).
    This also self-heals the audit_log schema before writing.
    """
    ensure_audit_log_table()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""INSERT INTO {_AUDIT_TABLE}
                (timestamp, action_type, case_id, user, details)
                VALUES (?, ?, ?, ?, ?)""",
            (ts, str(action_type or ""), str(case_id or ""), str(user or ""), str(details or ""))
        )
        conn.commit()
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Duplicate Detection
# -----------------------------------------------------------------------------
def detect_cosine_duplicates(df: pd.DataFrame, threshold: float = 0.85):
    if df is None or df.empty or "issue" not in df.columns:
        return []
    issues = df["issue"].fillna("").astype(str).tolist()
    if not issues:
        return []
    try:
        vectors = TfidfVectorizer().fit_transform(issues)
        sim = cosine_similarity(vectors)
    except Exception:
        return []

    duplicates = []
    n = len(issues)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                if sim[i, j] > float(threshold):
                    duplicates.append((df.iloc[i].get("id"), df.iloc[j].get("id")))
            except Exception:
                continue
    return duplicates

# -----------------------------------------------------------------------------
# Email Thread Grouping (heuristic)
# -----------------------------------------------------------------------------
def link_email_threads(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame().groupby([])  # harmless no-op

    def _thread_key(x: str) -> str:
        if pd.isna(x):
            return ""
        s = str(x)
        s = re.sub(r"^(Re:|Fwd:)\s*", "", s, flags=re.IGNORECASE)
        base = s.split("-")[0]
        return base.strip().lower()

    df = df.copy()
    src_col = "subject" if "subject" in df.columns else "issue"
    df["thread_id"] = df[src_col].apply(_thread_key)
    return df.groupby("thread_id")

# -----------------------------------------------------------------------------
# Lightweight Plugin Loader
# -----------------------------------------------------------------------------
def load_custom_plugins(path: str = "plugins/"):
    if not os.path.isdir(path):
        return
    for file in os.listdir(path):
        if not file.endswith(".py"):
            continue
        full = os.path.join(path, file)
        try:
            spec = importlib.util.spec_from_file_location(file[:-3], full)
            mod = importlib.util.module_from_spec(spec)
            if spec and spec.loader:
                spec.loader.exec_module(mod)
        except Exception as e:
            st.warning(f"Failed to load plugin {file}: {e}")

# -----------------------------------------------------------------------------
# WhatsApp (placeholder)
# -----------------------------------------------------------------------------
def send_whatsapp_message(phone: str, message: str) -> bool:
    try:
        url = os.getenv("WHATSAPP_API_URL", "https://api.twilio.com/...")  # replace with real
        token = os.getenv("WHATSAPP_API_TOKEN", "YOUR_TOKEN")
        payload = {"to": str(phone), "body": str(message)}
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        return 200 <= resp.status_code < 300
    except Exception as e:
        st.warning(f"WhatsApp send failed: {e}")
        return False
