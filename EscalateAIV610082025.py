# EscalateAIV610082025.py
# --------------------------------------------------------------------
# EscalateAI — Customer Escalation Prediction & Management Tool
# Updated per request:
# 1) "How this Dashboard Works" moved into a separate tab.
# 2) Added detailed explanations/docstrings and inline comments.
# 3) Removed top-line counts; show counts inside the colored status bars.
# --------------------------------------------------------------------

# ======================
# Imports & Environment
# ======================
# Standard library imports for OS, time, threading, hashing, DB, and email handling
import os
import re
import time
import datetime
import threading
import hashlib
import sqlite3
import smtplib
import requests
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import traceback

# Data & UI libraries
import pandas as pd
import numpy as np
import streamlit as st

# ML & NLP libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Environment variable loader
from dotenv import load_dotenv


# ==========================
# Optional Modules (Try/No-op)
# ==========================
# We try to import optional helper modules. If they're not present, we provide
# safe fallbacks so the app remains usable with core functionality.
if not os.path.exists("enhancements.py"):
    st.warning("⚠️ enhancements.py not found — some analytics/visuals may be unavailable.")
if not (os.path.exists("advanced_enhancements.py") or os.path.exists("advanced_enhancements V6.09.py")):
    st.warning("⚠️ advanced_enhancements.py not found — some advanced features may be unavailable.")

try:
    from enhancements import (
        schedule_weekly_retraining,
        render_analytics,
        show_feature_importance,
        is_duplicate,
        generate_pdf_report,
        render_sla_heatmap,
        apply_dark_mode,
        show_filter_summary,
        get_escalation_template,
        summarize_escalations,
    )
except Exception:
    # Fallbacks that keep the app running if enhancements.py is missing
    def render_analytics(): st.info("enhancements.render_analytics not available.")
    def show_feature_importance(*a, **k): st.info("enhancements.show_feature_importance not available.")
    def generate_pdf_report(): raise RuntimeError("enhancements.generate_pdf_report missing")
    def render_sla_heatmap(): st.info("enhancements.render_sla_heatmap not available.")
    def apply_dark_mode(): st.info("enhancements.apply_dark_mode not available.")
    def show_filter_summary(*a, **k): pass
    def summarize_escalations(): return "No enhancements summary."

try:
    from advanced_enhancements import (
        predict_resolution_eta,
        show_shap_explanation,
        detect_cosine_duplicates,
        link_email_threads,
        load_custom_plugins,
        send_whatsapp_message,
        generate_text_pdf,
        render_model_metrics,
        score_feedback_quality,
        validate_escalation_schema,
        log_escalation_action
    )
except Exception:
    # Safe fallbacks for advanced features
    def validate_escalation_schema(): pass
    def log_escalation_action(*a, **k): pass
    def load_custom_plugins(): pass
    def send_whatsapp_message(*a, **k): return False


# =================
# Analytics (Light)
# =================
def show_analytics_view():
    """
    Render a lightweight analytics page without requiring enhancements.py.
    Shows volume trend, severity distribution, sentiment distribution, and ageing buckets.
    """
    df = fetch_escalations()
    st.title("📊 Escalation Analytics")

    if df.empty:
        st.warning("⚠️ No escalation data available.")
        return

    # Ensure timestamp is datetime for proper grouping and age calc
    st.subheader("📈 Escalation Volume Over Time")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    trend = df.groupby(df['timestamp'].dt.date).size()
    st.line_chart(trend)

    st.subheader("🔥 Severity Distribution")
    st.bar_chart(df['severity'].value_counts())

    st.subheader("🧠 Sentiment Breakdown")
    st.bar_chart(df['sentiment'].value_counts())

    st.subheader("⏳ Ageing Buckets")
    df['age_days'] = (pd.Timestamp.now() - df['timestamp']).dt.days
    bins = [0, 3, 7, 14, 30, 90]
    labels = ["0–3d", "4–7d", "8–14d", "15–30d", "31–90d"]
    df['age_bucket'] = pd.cut(df['age_days'], bins=bins, labels=labels)
    st.bar_chart(df['age_bucket'].value_counts().sort_index())


# ==============
# Configuration
# ==============
load_dotenv()  # Reads .env to load credentials and settings

# Email, Teams & SMTP configuration
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS") or ""
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER", EMAIL_USER or "")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL", "")
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "🚨 EscalateAI Alert")

# SQLite DB location and ID prefix
DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-25"

# NLP analyzer for quick sentiment tagging
analyzer = SentimentIntensityAnalyzer()

# Keyword buckets used to infer urgency/category quickly
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge", "leak"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

# Concurrency guard for background email processor
processed_email_uids_lock = threading.Lock()
global_seen_hashes = set()  # Dedups incoming email content

# UI Colors used for status/severity chips
STATUS_COLORS = {"Open": "#FFA500", "In Progress": "#1E90FF", "Resolved": "#32CD32"}
SEVERITY_COLORS = {"critical": "#FF4500", "major": "#FF8C00", "minor": "#228B22"}
URGENCY_COLORS = {"high": "#DC143C", "normal": "#008000"}


# ==================
# Helper / DB Utils
# ==================
def summarize_issue_text(issue_text: str) -> str:
    """
    Create a short, safe summary of long issue text so cards stay readable.
    """
    clean_text = re.sub(r'\s+', ' ', issue_text or "").strip()
    return clean_text[:120] + "..." if len(clean_text) > 120 else clean_text


def get_next_escalation_id() -> str:
    """
    Generate a sequential ID like 'SESICE-25xxxxx' by looking up the latest numeric suffix.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
    last = cur.fetchone()
    conn.close()

    if last:
        last_id = last[0]
        # Strip prefix and parse the tail number
        last_num_str = last_id.replace(ESCALATION_PREFIX, "")
        try:
            last_num = int(last_num_str)
        except ValueError:
            last_num = 0
        next_num = last_num + 1
    else:
        next_num = 1

    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"


def ensure_schema():
    """
    Create or patch the 'escalations' table. Also attempts to add newer columns safely.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Base schema
        cur.execute('''
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                sentiment TEXT,
                urgency TEXT,
                severity TEXT,
                criticality TEXT,
                category TEXT,
                status TEXT,
                timestamp TEXT,
                action_taken TEXT,
                owner TEXT,
                owner_email TEXT,
                escalated TEXT,
                priority TEXT,
                likely_to_escalate TEXT,
                action_owner TEXT,
                status_update_date TEXT,
                user_feedback TEXT
            )
        ''')
        # Patch newer columns idempotently
        for col in ["owner_email", "status_update_date", "user_feedback", "likely_to_escalate", "action_owner", "priority"]:
            try:
                cur.execute(f"SELECT {col} FROM escalations LIMIT 1")
            except Exception:
                try:
                    cur.execute(f"ALTER TABLE escalations ADD COLUMN {col} TEXT")
                except Exception:
                    traceback.print_exc()
        conn.commit()
    except Exception:
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def generate_issue_hash(issue_text: str) -> str:
    """
    Normalize email content (remove headers/quotes/extra whitespace) and MD5 it.
    Used to deduplicate similar incoming emails.
    """
    patterns_to_remove = [
        r"[-]+[ ]*Forwarded message[ ]*[-]+",
        r"From:.*", r"Sent:.*", r"To:.*", r"Subject:.*",
        r">.*",  # quoted lines
        r"On .* wrote:",
        r"\n\s*\n"
    ]
    for pat in patterns_to_remove:
        issue_text = re.sub(pat, "", issue_text or "", flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', (issue_text or "").lower().strip())
    return hashlib.md5(clean_text.encode()).hexdigest()


def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category,
                      escalation_flag, likely_to_escalate="No", owner_email=""):
    """
    Insert a new escalation record into SQLite.

    Mapping notes:
    - 'escalation_flag' is stored in 'escalated' as Yes/No.
    - 'priority' becomes 'high' when severity is critical or urgency is high, else 'normal'.
    """
    ensure_schema()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    priority = "high" if str(severity).lower() == "critical" or str(urgency).lower() == "high" else "normal"

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO escalations (
                id, customer, issue, sentiment, urgency, severity, criticality, category,
                status, timestamp, action_taken, owner, owner_email, escalated, priority,
                likely_to_escalate, action_owner, status_update_date, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            new_id, customer, issue, sentiment, urgency, severity, criticality, category,
            "Open", now, "", "", owner_email or "", escalation_flag, priority,
            likely_to_escalate, "", "", ""
        ))
        conn.commit()
    except Exception as e:
        st.error(f"DB insert failed for {new_id}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_escalations() -> pd.DataFrame:
    """
    Read all escalations as a DataFrame. Always ensures the schema first.
    """
    ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception as e:
        st.error(f"Error reading escalations table: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def update_escalation_status(esc_id, status, action_taken, action_owner,
                             owner_email=None, feedback=None, sentiment=None,
                             criticality=None, notes=None):
    """
    Update common fields for an escalation and audit the change if possible.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        UPDATE escalations
        SET status = ?, action_taken = ?, action_owner = ?, status_update_date = ?,
            owner_email = ?, user_feedback = ?, sentiment = ?, criticality = ?
        WHERE id = ?
    ''', (
        status,
        action_taken,
        action_owner,
        datetime.datetime.now().isoformat(),
        owner_email,
        notes if notes is not None else feedback,
        sentiment,
        criticality,
        esc_id
    ))
    conn.commit()
    conn.close()

    # Optional audit log
    try:
        log_escalation_action("update_status", esc_id, action_owner or "system",
                              f"status={status}; action_taken={action_taken}")
    except Exception:
        pass


# ============
# Email Utils
# ============
def parse_emails():
    """
    Fetch UNSEEN emails via IMAP, normalize, and return unique (customer, issue summary) entries.
    Requires EMAIL_* env vars set in .env.
    """
    emails_out = []
    conn = None
    try:
        if not EMAIL_USER:
            st.warning("Email credentials not configured. Set EMAIL_USER/EMAIL_PASS in .env")
            return emails_out

        conn = imaplib.IMAP4_SSL(EMAIL_SERVER)
        conn.login(EMAIL_USER, EMAIL_PASS)
        conn.select("inbox")
        _, messages = conn.search(None, "UNSEEN")

        for num in messages[0].split():
            _, msg_data = conn.fetch(num, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg.get("Subject", ""))[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(errors='ignore')
                    from_ = msg.get("From", "unknown")

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode(errors='ignore')
                                except Exception:
                                    body = ""
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode(errors='ignore')
                        except Exception:
                            body = ""

                    full_text = f"{subject} - {body}"
                    hash_val = generate_issue_hash(full_text)
                    if hash_val not in global_seen_hashes:
                        global_seen_hashes.add(hash_val)
                        summary = summarize_issue_text(full_text)
                        emails_out.append({"customer": from_, "issue": summary})
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
    finally:
        if conn:
            try:
                conn.logout()
            except Exception:
                pass
    return emails_out


# ==========
# NLP/Tags
# ==========
def analyze_issue(issue_text: str):
    """
    Fast rules-based tagging:
    - Sentiment via VADER
    - Urgency via keyword scan
    - Category via first matching bucket
    - Severity derived from category; Criticality & Escalation flag from sentiment/urgency
    """
    scores = analyzer.polarity_scores(issue_text or "")
    compound = scores["compound"]
    if compound < -0.05:
        sentiment = "negative"
    elif compound > 0.05:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    text_lower = (issue_text or "").lower()
    urgency = "high" if any(word in text_lower for cat in NEGATIVE_KEYWORDS.values() for word in cat) else "normal"

    category = None
    for cat, keywords in NEGATIVE_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            category = cat
            break

    if category in ["safety", "technical"]:
        severity = "critical"
    elif category in ["support", "business"]:
        severity = "major"
    else:
        severity = "minor"

    criticality = "high" if (sentiment == "negative" and urgency == "high") else "medium"
    escalation_flag = "Yes" if (urgency == "high" or sentiment == "negative") else "No"
    return sentiment, urgency, severity, criticality, category or "other", escalation_flag


# =========
# ML Model
# =========
def train_model():
    """
    Train a small RandomForest on categorical features when enough data exists.
    If data is insufficient or target is single-class, returns None to fall back to rules.
    """
    df = fetch_escalations()
    if df.shape[0] < 20:
        return None

    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'likely_to_escalate'])
    if df.empty:
        return None

    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['likely_to_escalate'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    if y.nunique() < 2:
        return None

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def predict_escalation(model, sentiment, urgency, severity, criticality):
    """
    Predict "Yes"/"No" for likely_to_escalate using the model if present,
    else apply a simple rule (2 of 3 risk signals).
    """
    if model is None:
        risk_severity = str(severity).lower() in ["critical", "high"]
        risk_urgency = str(urgency).lower() in ["high", "immediate"]
        risk_sentiment = str(sentiment).lower() in ["negative", "very negative"]
        return "Yes" if (risk_severity + risk_urgency + risk_sentiment) >= 2 else "No"

    X_pred = pd.DataFrame([{
        f"sentiment_{sentiment}": 1,
        f"urgency_{urgency}": 1,
        f"severity_{severity}": 1,
        f"criticality_{criticality}": 1
    }])
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    pred = model.predict(X_pred)
    return "Yes" if int(pred[0]) == 1 else "No"


# =========
# Alerting
# =========
def send_alert(message: str, via: str = "email", recipient: str | None = None):
    """
    Send alert notifications via email or MS Teams webhook.
    """
    if via == "email":
        try:
            if not ALERT_RECIPIENT and not recipient:
                st.warning("No email recipient configured.")
                return
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['Subject'] = EMAIL_SUBJECT
            msg['From'] = EMAIL_USER or "no-reply@escalateai"
            msg['To'] = recipient if recipient else ALERT_RECIPIENT
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                if EMAIL_USER and EMAIL_PASS:
                    server.login(EMAIL_USER, EMAIL_PASS)
                server.sendmail(msg['From'], [msg['To']], msg.as_string())
        except Exception as e:
            st.error(f"Email alert failed: {e}")

    elif via == "teams":
        try:
            if not TEAMS_WEBHOOK:
                st.error("MS Teams webhook URL is not configured.")
                return
            response = requests.post(
                TEAMS_WEBHOOK,
                json={"text": message},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                st.error(f"Teams alert failed: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Teams alert failed: {e}")


# ===========================
# Background Email Polling
# ===========================
def email_polling_job():
    """
    Background loop: checks inbox every 60s, analyzes new emails, inserts new cases.
    Note: Runs in a daemon thread so it doesn't block the UI.
    """
    while True:
        model = train_model()
        emails = parse_emails()
        with processed_email_uids_lock:
            for e in emails:
                issue = e["issue"]
                customer = e["customer"]
                sent, urg, sev, crit, cat, esc = analyze_issue(issue)
                likely_to_escalate = predict_escalation(model, sent, urg, sev, crit)
                insert_escalation(customer, issue, sent, urg, sev, crit, cat, esc, likely_to_escalate)
        time.sleep(60)


# ================
# Streamlit Setup
# ================
st.set_page_config(page_title="Escalation Management", layout="wide")

# Ensure DB/table exists and optional validators run
ensure_schema()
try:
    validate_escalation_schema()
except Exception:
    pass

# Load optional plugins gracefully
try:
    load_custom_plugins()
except Exception:
    pass

# Header / Branding
st.markdown(
    """
    <style>
    header h1 { margin: 0; padding-left: 20px; }
    </style>
    <header>
        <div>
            <h1>🚨 EscalateAI – AI Based Customer Escalation Prediction & Management Tool</h1>
        </div>
    </header>
    """,
    unsafe_allow_html=True
)


# =====================
# Sidebar & Navigation
# =====================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Main Dashboard",
    "📈 Analytics",
    "🔥 SLA Heatmap",
    "🧠 Enhancements",
    "⚙️ Admin Tools"
])

# Email ingestion
st.sidebar.markdown("### 📩 Email Integration")
if st.sidebar.button("Fetch Emails"):
    emails = parse_emails()
    model = train_model()
    for e in emails:
        issue, customer = e["issue"], e["customer"]
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        likely_to_escalate = predict_escalation(model, sentiment, urgency, severity, criticality)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, likely_to_escalate)
    st.sidebar.success(f"✅ {len(emails)} emails processed")

# Upload Excel
st.sidebar.header("📁 Upload Escalation Sheet")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded_file:
    try:
        df_excel = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ Excel file loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to read Excel file: {e}")
        st.stop()

    required_columns = ["Customer", "Issue"]
    missing_cols = [c for c in required_columns if c not in df_excel.columns]
    if missing_cols:
        st.sidebar.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    if st.sidebar.button("🔍 Analyze & Insert"):
        model = train_model()
        processed_count = 0
        for idx, row in df_excel.iterrows():
            issue = str(row.get("Issue", "")).strip()
            customer = str(row.get("Customer", "Unknown")).strip()
            if not issue:
                st.warning(f"⚠️ Row {idx + 1} skipped: empty issue text.")
                continue
            issue_summary = summarize_issue_text(issue)
            sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
            likely_to_escalate = predict_escalation(model, sentiment, urgency, severity, criticality)
            insert_escalation(customer, issue_summary, sentiment, urgency, severity, criticality, category,
                              escalation_flag, likely_to_escalate)
            processed_count += 1
        st.sidebar.success(f"🎯 {processed_count} rows processed successfully.")

# SLA checker
st.sidebar.markdown("### ⏰ SLA Monitor")
if st.sidebar.button("Trigger SLA Check"):
    df_tmp = fetch_escalations()
    if not df_tmp.empty:
        df_tmp['timestamp'] = pd.to_datetime(df_tmp['timestamp'], errors='coerce')
        breaches = df_tmp[(df_tmp['status'].str.title() != 'Resolved') &
                          (df_tmp['priority'].str.lower() == 'high') &
                          ((datetime.datetime.now() - df_tmp['timestamp']) > datetime.timedelta(minutes=10))]
        if not breaches.empty:
            alert_msg = f"🚨 SLA breach for {len(breaches)} case(s)!"
            send_alert(alert_msg, via="teams")
            send_alert(alert_msg, via="email")
            st.sidebar.success("✅ Alerts sent")
        else:
            st.sidebar.info("All SLAs healthy")
    else:
        st.sidebar.info("No data yet.")

# Sidebar filters (moved as requested)
st.sidebar.markdown("### 🔍 Escalation Filters")
status_opt    = st.sidebar.selectbox("Status",   ["All", "Open", "In Progress", "Resolved"], index=0)
severity_opt  = st.sidebar.selectbox("Severity", ["All", "minor", "major", "critical"], index=0)
sentiment_opt = st.sidebar.selectbox("Sentiment",["All", "positive", "neutral", "negative"], index=0)
category_opt  = st.sidebar.selectbox("Category", ["All", "technical", "support", "dissatisfaction", "safety", "business", "other"], index=0)

# Manual alerts
st.sidebar.markdown("### 🔔 Manual Notifications")
manual_msg = st.sidebar.text_area("Compose Alert", "🚨 Test alert from EscalateAI")
col_send1, col_send2 = st.sidebar.columns(2)
with col_send1:
    if st.button("Send MS Teams"):
        send_alert(manual_msg, via="teams")
        st.sidebar.success("✅ MS Teams alert sent")
with col_send2:
    if st.button("Send Email"):
        send_alert(manual_msg, via="email")
        st.sidebar.success("✅ Email alert sent")

# WhatsApp
st.sidebar.markdown("### 📲 WhatsApp Alerts")
status_check = st.sidebar.selectbox("Case Status", ["Open", "In Progress", "Resolved"])
df_all_for_wa = fetch_escalations()
if status_check == "Resolved":
    df_resolved = df_all_for_wa[df_all_for_wa["status"].str.strip().str.title() == "Resolved"]
    if not df_resolved.empty:
        escalation_id = st.sidebar.selectbox(
            "🔢 Select Resolved Escalation ID",
            df_resolved["id"].astype(str).tolist()
        )
        phone = st.sidebar.text_input("📞 Phone Number", "+91", help="Include country code (e.g., +91)")
        w_msg = st.sidebar.text_area("📨 Message", f"Your issue with ID {escalation_id} has been resolved. Thank you!")
        if st.sidebar.button("Send WhatsApp"):
            try:
                ok = send_whatsapp_message(phone, w_msg) if callable(send_whatsapp_message) else False
                if ok:
                    st.sidebar.success(f"✅ WhatsApp sent to {phone} for Escalation ID {escalation_id}")
                else:
                    st.sidebar.error("❌ WhatsApp API returned failure")
            except Exception as e:
                st.sidebar.error(f"❌ WhatsApp send failed: {e}")
    else:
        st.sidebar.warning("No resolved escalations found.")
else:
    st.sidebar.info("WhatsApp alerts are only available for 'Resolved' cases.")

# Downloads
st.sidebar.markdown("### 📤 Downloads")
col_dl1, col_dl2 = st.sidebar.columns(2)
with col_dl1:
    if st.button("⬇️ All Complaints"):
        csv = fetch_escalations().to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")
with col_dl2:
    if st.button("⬇️ Escalated Only"):
        df_esc_only = fetch_escalations()
        if not df_esc_only.empty:
            df_esc_only = df_esc_only[df_esc_only["escalated"].str.lower() == "yes"]
        if df_esc_only.empty:
            st.info("No escalated cases.")
        else:
            out_path = "escalated_cases.xlsx"
            with pd.ExcelWriter(out_path) as writer:
                df_esc_only.to_excel(writer, index=False)
            with open(out_path, "rb") as f:
                st.download_button("Download Excel", f, file_name=out_path,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Behavior toggles
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 10, 60, 30)
compact_mode = st.sidebar.checkbox("📱 Compact Mode", value=False)
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
if st.sidebar.button("🔁 Manual Refresh"):
    st.rerun()

# Dark mode & summary
if st.sidebar.checkbox("🌙 Dark Mode"):
    try:
        apply_dark_mode()
    except Exception:
        pass

st.sidebar.subheader("🧠 AI Assistant Summary")
try:
    st.sidebar.write(summarize_escalations())
except Exception:
    st.sidebar.write("Summary unavailable.")

if st.sidebar.button("📄 Generate PDF Report"):
    try:
        generate_pdf_report()
        st.sidebar.success("PDF report generated as report.pdf")
    except Exception as e:
        st.sidebar.error(f"PDF generation failed: {e}")


# ===============================
# Utility: Search filtering
# ===============================
def filter_df_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Simple full-text search across common fields (id, customer, issue, owner, etc.)
    """
    if not query:
        return df
    q = str(query).strip().lower()
    if df.empty:
        return df
    cols = ['id', 'customer', 'issue', 'owner', 'action_owner', 'owner_email',
            'category', 'severity', 'sentiment', 'status']
    present = [c for c in cols if c in df.columns]
    combined = df[present].astype(str).apply(lambda s: s.str.lower()).agg(' '.join, axis=1)
    return df[combined.str.contains(q, na=False, regex=False)]


# ===============================
# Main Page Routing
# ===============================
if page == "📊 Main Dashboard":
    # Load data and apply sidebar filters
    df_all = fetch_escalations()
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')

    # Apply sidebar filters uniformly
    filtered_df = df_all.copy()
    if status_opt != "All":
        filtered_df = filtered_df[filtered_df["status"].str.strip().str.title() == status_opt]
    if severity_opt != "All":
        filtered_df = filtered_df[filtered_df["severity"].str.lower() == severity_opt.lower()]
    if sentiment_opt != "All":
        filtered_df = filtered_df[filtered_df["sentiment"].str.lower() == sentiment_opt.lower()]
    if category_opt != "All":
        filtered_df = filtered_df[filtered_df["category"].str.lower() == category_opt.lower()]

    # SLA red banner (computed post-filter)
    breaches_banner = filtered_df[(filtered_df['status'].str.title() != 'Resolved') &
                                  (filtered_df['priority'].str.lower() == 'high') &
                                  ((datetime.datetime.now() - filtered_df['timestamp']) > datetime.timedelta(minutes=10))]
    if not breaches_banner.empty:
        st.markdown(
            f"<div style='background:#dc3545;padding:8px;border-radius:5px;color:white;text-align:center;'>"
            f"<strong>🚨 {len(breaches_banner)} SLA Breach(s) Detected</strong></div>",
            unsafe_allow_html=True
        )

    # ---------- Tabs (includes new Help tab) ----------
    tabs = st.tabs(["ℹ️ How this Dashboard Works", "🗃️ All", "🚩 Likely to Escalate", "🔁 Feedback & Retraining", "📊 Analytics"])

    # --------------------- Tab 0: Help ---------------------
    with tabs[0]:
        st.subheader("ℹ️ How this Dashboard Works")
        st.markdown("""
**What you see**
- **Kanban Board** split into **Open**, **In Progress**, **Resolved** columns.  
- Cards show **Severity**, **Urgency**, **Criticality**, **Category**, **Sentiment**, **Age**, and a **Likely to Escalate** badge.

**How "Likely to Escalate" is computed**
- If a trained model exists, it predicts using: `sentiment`, `urgency`, `severity`, `criticality`.  
- Otherwise a fallback rule returns **Yes** if at least **two** conditions hold:
  - Severity is *critical/high*
  - Urgency is *high/immediate*
  - Sentiment is *negative/very negative*

**IDs & Priority**
- IDs look like **SESICE-25xxxxx** (auto-generated sequentially).  
- **Priority** = *high* if **Severity=critical** or **Urgency=high**, else *normal*.  
- SLA warnings: high-priority unresolved cases older than **10 minutes**.

**Actions on a card**
- **✔️ Resolved** — marks resolved and notifies owner via Email/Teams.  
- **🚀 To N+1** — forwards the case to the typed email (escalation).  
- **💾 Save Changes** — updates Status, Action Taken, Owner, Owner Email.

**Color legend**
- Severity: critical=red, major=orange, minor=green  
- Urgency: high=red, normal=green  
- Likely badge: red if **Yes**, grey if **No**
        """)

    # --------------------- Tab 1: All ---------------------
    with tabs[1]:
        st.subheader("📊 Escalation Kanban Board — All Cases")
        # Search bar for "All"
        search_all = st.text_input("🔍 Search cases", placeholder="Search by ID, customer, issue, owner, email, status…")
        df_view = filter_df_by_query(filtered_df.copy(), search_all)

        # Normalize status labels used in column matching
        df_view["status"] = df_view["status"].fillna("Open").str.strip().str.title()
        counts = df_view['status'].value_counts()

        # 3 columns for Kanban — counts now inside colored headers (no separate count line)
        col1, col2, col3 = st.columns(3)
        status_columns = {"Open": col1, "In Progress": col2, "Resolved": col3}

        for status_name, col in status_columns.items():
            with col:
                # Show count in the header bar (requested change)
                count_here = int(counts.get(status_name, 0))
                col.markdown(
                    f"<h3 style='background-color:{STATUS_COLORS[status_name]};color:white;padding:8px;"
                    f"border-radius:5px;text-align:center;'>{status_name} &nbsp; <span style='opacity:0.9;'>({count_here})</span></h3>",
                    unsafe_allow_html=True
                )
                bucket = df_view[df_view["status"] == status_name]
                for _, row in bucket.iterrows():
                    try:
                        summary = summarize_issue_text(row.get('issue', ''))

                        # Compute likely_to_escalate robustly at render time
                        model = train_model()
                        sentiment = (row.get("sentiment") or "neutral").lower()
                        urgency = (row.get("urgency") or "normal").lower()
                        severity = (row.get("severity") or "minor").lower()
                        criticality = (row.get("criticality") or "medium").lower()
                        likely_to_escalate = predict_escalation(model, sentiment, urgency, severity, criticality)

                        flag = "🚩" if likely_to_escalate == 'Yes' else ""
                        expander_label = f"{row.get('id', 'N/A')} - {row.get('customer', 'Unknown')} {flag} – {summary}"
                        prefix = f"case_{row.get('id', 'N/A')}"

                        # Visual color chips
                        header_color = SEVERITY_COLORS.get(severity, "#7f8c8d")
                        urgency_color = URGENCY_COLORS.get(urgency, "#7f8c8d")
                        sentiment_cap = (row.get("sentiment") or "neutral").capitalize()
                        sentiment_color = {"Negative": "#e74c3c", "Positive": "#2ecc71", "Neutral": "#f39c12"}.get(sentiment_cap, "#7f8c8d")
                        escalated_color = "#c0392b" if likely_to_escalate == "Yes" else "#7f8c8d"
                        category = (row.get("category") or "other").capitalize()
                        criticality_cap = (row.get("criticality") or "medium").capitalize()

                        # Age calculation
                        try:
                            ts = pd.to_datetime(row.get("timestamp"))
                            now = datetime.datetime.now()
                            delta = now - ts
                            days = delta.days
                            hours, remainder = divmod(delta.seconds, 3600)
                            minutes, _ = divmod(remainder, 60)
                            ageing_str = f"{days}d {hours}h {minutes}m"
                            total_hours = delta.total_seconds() / 3600
                            ageing_color = "#2ecc71" if total_hours < 12 else "#f39c12" if total_hours < 24 else "#e74c3c"
                        except Exception:
                            ageing_str = "N/A"
                            ageing_color = "#7f8c8d"

                        with st.expander(f"📂 {expander_label}", expanded=False):
                            # Action row (resolve/escalate)
                            if not compact_mode:
                                colA, colB, colC, colD = st.columns([1, 2, 2, 1])
                                with colA:
                                    # Age chip
                                    st.markdown(
                                        f"<div style='background-color:{ageing_color};padding:6px;border-radius:5px;"
                                        f"color:white;text-align:center'>Age: {ageing_str}</div>",
                                        unsafe_allow_html=True
                                    )
                                with colB:
                                    # Resolve button
                                    if st.button("✔️ Resolved", key=f"{prefix}_resolved"):
                                        owner_email = row.get("owner_email", EMAIL_USER)
                                        update_escalation_status(row['id'], "Resolved",
                                                                 row.get("action_taken", ""),
                                                                 row.get("owner", ""),
                                                                 owner_email)
                                        if owner_email:
                                            send_alert("Case marked as resolved.", via="email", recipient=owner_email)
                                        send_alert("Case marked as resolved.", via="teams")
                                with colC:
                                    # N+1 escalation email input
                                    n1_email = st.text_input("N+1 Email", key=f"{prefix}_n1email")
                                with colD:
                                    # Escalate button
                                    if st.button("🚀 To N+1", key=f"{prefix}_n1btn"):
                                        update_escalation_status(row['id'], row.get("status", "Open"),
                                                                 row.get("action_taken", ""),
                                                                 row.get("owner", ""),
                                                                 n1_email)
                                        if n1_email:
                                            send_alert("Case escalated to N+1.", via="email", recipient=n1_email)
                                        send_alert("Case escalated to N+1.", via="teams")

                            # Metadata chips
                            row1_col1, row1_col2, row1_col3 = st.columns(3)
                            with row1_col1:
                                st.markdown("**📛 Severity**")
                                st.markdown(
                                    f"<div style='background-color:{header_color};padding:6px;border-radius:5px;color:white;text-align:center'>{severity.capitalize()}</div>",
                                    unsafe_allow_html=True
                                )
                            with row1_col2:
                                st.markdown("**⚡ Urgency**")
                                st.markdown(
                                    f"<div style='background-color:{urgency_color};padding:6px;border-radius:5px;color:white;text-align:center'>{urgency.capitalize()}</div>",
                                    unsafe_allow_html=True
                                )
                            with row1_col3:
                                st.markdown("**🎯 Criticality**")
                                st.markdown(
                                    f"<div style='background-color:#8e44ad;padding:6px;border-radius:5px;color:white;text-align:center'>{criticality_cap}</div>",
                                    unsafe_allow_html=True
                                )

                            row2_col1, row2_col2, row2_col3 = st.columns(3)
                            with row2_col1:
                                st.markdown("**📂 Category**")
                                st.markdown(
                                    f"<div style='background-color:#16a085;padding:6px;border-radius:5px;color:white;text-align:center'>{category}</div>",
                                    unsafe_allow_html=True
                                )
                            with row2_col2:
                                st.markdown("**💬 Sentiment**")
                                st.markdown(
                                    f"<div style='background-color:{sentiment_color};padding:6px;border-radius:5px;color:white;text-align:center'>{sentiment_cap}</div>",
                                    unsafe_allow_html=True
                                )
                            with row2_col3:
                                st.markdown("**📈 Likely to Escalate**")
                                st.markdown(
                                    f"<div style='background-color:{escalated_color};padding:6px;border-radius:5px;color:white;text-align:center'>{likely_to_escalate}</div>",
                                    unsafe_allow_html=True
                                )

                            # Editable fields row
                            edit_row1_col1, edit_row1_col2 = st.columns(2)
                            with edit_row1_col1:
                                current_status = (row.get("status") or "Open").strip().str.title()
                                new_status = st.selectbox(
                                    "Update Status", ["Open", "In Progress", "Resolved"],
                                    index=["Open", "In Progress", "Resolved"].index(current_status) if current_status in ["Open", "In Progress", "Resolved"] else 0,
                                    key=f"{prefix}_status"
                                )
                            with edit_row1_col2:
                                new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"{prefix}_action")

                            edit_row2_col1, edit_row2_col2 = st.columns(2)
                            with edit_row2_col1:
                                new_owner = st.text_input("Owner", row.get("owner", ""), key=f"{prefix}_owner")
                            with edit_row2_col2:
                                new_owner_email = st.text_input("Owner Email", row.get("owner_email", ""), key=f"{prefix}_email")

                            if st.button("💾 Save Changes", key=f"{prefix}_save"):
                                update_escalation_status(row['id'], new_status, new_action, new_owner, new_owner_email)
                                st.success("Escalation updated.")
                                notification_message = f"""
🔔 Hello {new_owner or 'Owner'},
The escalation case #{row['id']} assigned to you has been updated:
• Status: {new_status}
• Action Taken: {new_action}
• Category: {category}
• Severity: {severity.capitalize()}
• Urgency: {urgency.capitalize()}
• Sentiment: {sentiment_cap}
Please review the updates on the EscalateAI dashboard.
                                """.strip()
                                if new_owner_email:
                                    send_alert(notification_message, via="email", recipient=new_owner_email)
                                send_alert(notification_message, via="teams")
                    except Exception as e:
                        st.error(f"Error rendering case #{row.get('id', 'Unknown')}: {e}")

    # ----------------- Tab 2: Likely to Escalate -----------------
    with tabs[2]:
        st.subheader("🚩 Likely to Escalate")
        # Build a likely-only view (model or rules), then search inside it
        df_le = filtered_df.copy()
        if not df_le.empty:
            model = train_model()
            def _predict_row(row):
                return predict_escalation(
                    model,
                    (row.get("sentiment") or "neutral").lower(),
                    (row.get("urgency") or "normal").lower(),
                    (row.get("severity") or "minor").lower(),
                    (row.get("criticality") or "medium").lower()
                )
            df_le["likely_calc"] = df_le.apply(_predict_row, axis=1)
            df_le = df_le[df_le["likely_calc"] == "Yes"]

        search_le = st.text_input("🔍 Search likely to escalate", placeholder="Search by ID, customer, issue, owner, email, status…")
        df_le = filter_df_by_query(df_le, search_le)

        st.markdown(f"**Cases predicted to escalate:** {len(df_le)}")
        st.dataframe(df_le.sort_values(by="timestamp", ascending=False), use_container_width=True)

    # ----------------- Tab 3: Feedback & Retraining -----------------
    with tabs[3]:
        st.subheader("🔁 Feedback & Retraining")
        df_fb = fetch_escalations()
        if not df_fb.empty:
            df_fb = df_fb[df_fb["likely_to_escalate"].notnull()]
            for _, row in df_fb.iterrows():
                with st.expander(f"🆔 {row['id']}"):
                    fb   = st.selectbox("Escalation Accuracy", ["Correct", "Incorrect"], key=f"fb_{row['id']}")
                    sent = st.selectbox("Sentiment", ["positive", "neutral", "negative"], key=f"sent_{row['id']}")
                    crit = st.selectbox("Criticality", ["low", "medium", "high", "urgent"], key=f"crit_{row['id']}")
                    notes= st.text_area("Notes", key=f"note_{row['id']}")
                    if st.button("Submit", key=f"btn_{row['id']}"):
                        owner_email = row.get("owner_email", EMAIL_USER)
                        update_escalation_status(row['id'], row.get("status", "Open"),
                                                 row.get("action_taken", ""), row.get("owner", ""),
                                                 owner_email, notes=notes, sentiment=sent, criticality=crit)
                        if owner_email:
                            send_alert("Feedback recorded on your case.", via="email", recipient=owner_email)
                        st.success("Feedback saved.")
        if st.button("🔁 Retrain Model"):
            st.info("Retraining model with feedback...")
            model = train_model()
            if model:
                st.success("Model retrained successfully.")
                try:
                    show_feature_importance(model)
                except Exception:
                    pass
            else:
                st.warning("Not enough data to retrain model.")

    # ----------------- Tab 4: Analytics -----------------
    with tabs[4]:
        st.subheader("📊 Escalation Analytics (Quick View)")
        try:
            render_analytics()
        except Exception as e:
            st.info("Analytics module not fully configured.")
            st.exception(e)

elif page == "🔥 SLA Heatmap":
    st.subheader("🔥 SLA Heatmap")
    try:
        render_sla_heatmap()
    except Exception as e:
        st.error(f"❌ SLA Heatmap failed to render: {type(e).__name__}: {str(e)}")

elif page == "🧠 Enhancements":
    try:
        from enhancement_dashboard import show_enhancement_dashboard
        show_enhancement_dashboard()
    except Exception as e:
        st.info("Enhancement dashboard not available.")
        st.exception(e)

elif page == "📈 Analytics":
    try:
        show_analytics_view()
    except Exception as e:
        st.error("❌ Failed to load analytics view.")
        st.exception(e)

elif page == "⚙️ Admin Tools":
    # Kept inline for clarity; could be moved out if you prefer.
    def show_admin_panel():
        """
        Admin page: validate schema, view audit log, write manual audit entries.
        """
        import sqlite3
        st.title("⚙️ Admin Tools")
        if st.button("🔍 Validate DB Schema"):
            try:
                validate_escalation_schema()
                st.success("✅ Schema validated and healed.")
            except Exception as e:
                st.error(f"❌ Schema validation failed: {e}")

        st.subheader("📄 Audit Log Preview")
        try:
            log_escalation_action("init", "N/A", "system", "Initializing audit log table")
            conn = sqlite3.connect("escalations.db")
            df = pd.read_sql("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100", conn)
            conn.close()
            st.dataframe(df)
        except Exception as e:
            st.warning("⚠️ Audit log not available.")
            st.exception(e)

        st.subheader("📝 Manual Audit Entry")
        with st.form("manual_log"):
            action = st.text_input("Action Type")
            case_id = st.text_input("Case ID")
            user = st.text_input("User")
            details = st.text_area("Details")
            submitted = st.form_submit_button("Log Action")
            if submitted:
                try:
                    log_escalation_action(action, case_id, user, details)
                    st.success("✅ Action logged.")
                except Exception as e:
                    st.error(f"❌ Failed to log action: {e}")

    try:
        show_admin_panel()
    except Exception as e:
        st.info("Admin tools not available.")
        st.exception(e)


# --------------------------
# Background Threads (once)
# --------------------------
if 'email_thread' not in st.session_state:
    # Start the background email polling daemon
    email_thread = threading.Thread(target=email_polling_job, daemon=True)
    email_thread.start()
    st.session_state['email_thread'] = email_thread


# --------------------------
# Daily Email Scheduler
# --------------------------
def send_daily_escalation_email():
    """
    Compose and email a daily Excel of likely-to-escalate cases with counts by status.
    """
    df = fetch_escalations()
    df_esc = df[df["likely_to_escalate"].str.lower() == "yes"] if not df.empty else df
    if df_esc.empty:
        return
    file_path = "daily_escalated_cases.xlsx"
    df_esc.to_excel(file_path, index=False)

    summary = f"""
🔔 Daily Escalation Summary – {datetime.datetime.now().strftime('%Y-%m-%d')}
Total Likely to Escalate Cases: {len(df_esc)}
Open: {df_esc[df_esc['status'].str.strip().str.title() == 'Open'].shape[0]}
In Progress: {df_esc[df_esc['status'].str.strip().str.title() == 'In Progress'].shape[0]}
Resolved: {df_esc[df_esc['status'].str.strip().str.title() == 'Resolved'].shape[0]}
Please find the attached Excel file for full details.
""".strip()

    try:
        msg = MIMEMultipart()
        msg['Subject'] = "📊 Daily Escalated Cases Report"
        msg['From'] = EMAIL_USER or "no-reply@escalateai"
        msg['To'] = ALERT_RECIPIENT or (EMAIL_USER or "")
        msg.attach(MIMEText(summary, 'plain'))
        with open(file_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{file_path}"')
            msg.attach(part)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            if EMAIL_USER and EMAIL_PASS:
                server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
    except Exception as e:
        print(f"❌ Failed to send daily email: {e}")


# Light scheduler that triggers once a day at 09:00
import schedule
import time as time_module
def schedule_daily_email():
    schedule.every().day.at("09:00").do(send_daily_escalation_email)
    def run_scheduler():
        while True:
            schedule.run_pending()
            time_module.sleep(60)
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()

if 'daily_email_thread' not in st.session_state:
    schedule_daily_email()
    st.session_state['daily_email_thread'] = True
