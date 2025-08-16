# EscalateAIV610082025.py
# --------------------------------------------------------------------
# EscalateAI — Customer Escalation Prediction & Management Tool
# Updates in this version:
# • Help tab moved to LAST in the Main Dashboard.
# • Sidebar item renamed to "📈 Advanced Analytics" (routing updated).
# • Main Dashboard tab renamed to "📊 Summary Analytics" to avoid name clash.
# • Bullet-proof duplicate detection:
#     - DB columns: issue_hash, duplicate_of; table: processed_hashes
#     - Exact hash match + TF-IDF cosine similarity + difflib fallback
#     - De-duping applied to Email fetch, Excel ingestion, and background polling
#     - Duplicates are skipped (not inserted) and audit-logged
# • Counts stay inside status bars (no duplication).
# --------------------------------------------------------------------

# ======================
# Imports & Environment
# ======================
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

import pandas as pd
import numpy as np
import streamlit as st

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# For duplicate detection (optional; we also have a difflib fallback)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _TFIDF_AVAILABLE = True
except Exception:
    _TFIDF_AVAILABLE = False

import difflib

from dotenv import load_dotenv

# ==========================
# Optional Modules (Try/No-op)
# ==========================
if not os.path.exists("enhancements.py"):
    st.warning("⚠️ enhancements.py not found — some analytics/visuals may be unavailable.")
if not (os.path.exists("advanced_enhancements.py") or os.path.exists("advanced_enhancements V6.09.py")):
    st.warning("⚠️ advanced_enhancements.py not found — some advanced features may be unavailable.")

try:
    from enhancements import (
        schedule_weekly_retraining,
        render_analytics,
        show_feature_importance,
        is_duplicate as _is_duplicate_ext,  # not used now; keeping for BC
        generate_pdf_report,
        render_sla_heatmap,
        apply_dark_mode,
        show_filter_summary,
        get_escalation_template,
        summarize_escalations,
    )
except Exception:
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
        detect_cosine_duplicates,  # not used; we provide built-in robust dedup
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
    def validate_escalation_schema(): pass
    def log_escalation_action(*a, **k): pass
    def load_custom_plugins(): pass
    def send_whatsapp_message(*a, **k): return False

# =================
# Analytics (Light)
# =================
def show_analytics_view():
    """Lightweight analytics page (volume trend, severity/sentiment, ageing)."""
    df = fetch_escalations()
    st.title("📊 Escalation Analytics")
    if df.empty:
        st.warning("⚠️ No escalation data available.")
        return
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
load_dotenv()

EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS") or ""

SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER", EMAIL_USER or "")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL", "")
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "🚨 EscalateAI Alert")

DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-25"

analyzer = SentimentIntensityAnalyzer()

NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge", "leak"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

processed_email_uids_lock = threading.Lock()
global_seen_hashes = set()

STATUS_COLORS = {"Open": "#FFA500", "In Progress": "#1E90FF", "Resolved": "#32CD32"}
SEVERITY_COLORS = {"critical": "#FF4500", "major": "#FF8C00", "minor": "#228B22"}
URGENCY_COLORS = {"high": "#DC143C", "normal": "#008000"}

# ==================
# Helper / DB Utils
# ==================
def summarize_issue_text(issue_text: str) -> str:
    """Create a short summary to keep cards readable."""
    clean_text = re.sub(r'\s+', ' ', issue_text or "").strip()
    return clean_text[:120] + "..." if len(clean_text) > 120 else clean_text

def _normalize_text(t: str) -> str:
    """Normalization used for hashing & similarity."""
    if not t:
        return ""
    patterns_to_remove = [
        r"[-]+[ ]*Forwarded message[ ]*[-]+",
        r"From:.*", r"Sent:.*", r"To:.*", r"Subject:.*",
        r">.*", r"On .* wrote:",
    ]
    for pat in patterns_to_remove:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def generate_issue_hash(issue_text: str) -> str:
    """Stable hash on normalized text."""
    clean_text = _normalize_text(issue_text or "")
    return hashlib.md5(clean_text.encode()).hexdigest()

def get_next_escalation_id() -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
    last = cur.fetchone()
    conn.close()
    if last:
        last_id = last[0]
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
    """Create/patch main table + dedup helper table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Main table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                issue_hash TEXT,
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
                user_feedback TEXT,
                duplicate_of TEXT
            )
        ''')
        # Add missing columns idempotently
        for col in ["issue_hash", "duplicate_of", "owner_email", "status_update_date",
                    "user_feedback", "likely_to_escalate", "action_owner", "priority"]:
            try:
                cur.execute(f"SELECT {col} FROM escalations LIMIT 1")
            except Exception:
                try:
                    cur.execute(f"ALTER TABLE escalations ADD COLUMN {col} TEXT")
                except Exception:
                    traceback.print_exc()

        # Processed hashes ledger (persists across restarts)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS processed_hashes (
                hash TEXT PRIMARY KEY,
                first_seen TEXT
            )
        ''')

        conn.commit()
    except Exception:
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _processed_hash_exists(h: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM processed_hashes WHERE hash=? LIMIT 1", (h,))
    row = cur.fetchone()
    conn.close()
    return row is not None

def _mark_processed_hash(h: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO processed_hashes (hash, first_seen) VALUES (?, ?)",
                    (h, datetime.datetime.now().isoformat()))
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def fetch_escalations() -> pd.DataFrame:
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

# ===========================
# Duplicate Detection (Core)
# ===========================
def _cosine_sim(a: str, b: str) -> float:
    """Cosine similarity with TF-IDF (1–2 grams). Fallback to difflib if TF-IDF unavailable."""
    a, b = _normalize_text(a), _normalize_text(b)
    if not a or not b:
        return 0.0
    if _TFIDF_AVAILABLE:
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            X = vec.fit_transform([a, b])
            sim = float(cosine_similarity(X[0], X[1]))
            return sim
        except Exception:
            pass
    # Fallback
    return difflib.SequenceMatcher(None, a, b).ratio()

def find_duplicate(issue_text: str, customer: str | None = None,
                   days_window: int = 180,
                   cosine_threshold: float = 0.88,
                   difflib_threshold: float = 0.92):
    """
    Returns (is_duplicate: bool, duplicate_id: str|None, score: float).

    Strategy:
      1) Exact hash match on normalized text.
      2) Cosine similarity (TF-IDF 1–2 grams) over recent records (<= days_window).
      3) difflib ratio as a strict fallback.
    """
    ensure_schema()
    text_norm = _normalize_text(issue_text or "")
    if not text_norm:
        return (False, None, 0.0)

    h = hashlib.md5(text_norm.encode()).hexdigest()

    # 1) Exact hash match first
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, issue FROM escalations WHERE issue_hash = ? LIMIT 1", (h,))
        row = cur.fetchone()
        if row:
            return (True, row[0], 1.0)

        # 2) Candidate pool (recent N days)
        df = pd.read_sql("SELECT id, issue, customer, timestamp FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return (False, None, 0.0)

    # Filter to time window if timestamps exist
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_window)
        df = df[df['timestamp'].isna() | (df['timestamp'] >= cutoff)]
    except Exception:
        pass

    # Optional: prioritize same customer
    if customer and 'customer' in df.columns:
        df_same = df[df['customer'].astype(str).str.lower() == str(customer).lower()]
        df_pool = df_same if not df_same.empty else df
    else:
        df_pool = df

    # Compute similarities and take the best
    best_id, best_score = None, 0.0
    for _, r in df_pool.iterrows():
        other = r.get('issue', '')
        score = _cosine_sim(issue_text, other)
        if score > best_score:
            best_score, best_id = score, r.get('id', None)

    # Threshold decision
    if _TFIDF_AVAILABLE and best_score >= cosine_threshold:
        return (True, best_id, best_score)
    elif (not _TFIDF_AVAILABLE) and best_score >= difflib_threshold:
        return (True, best_id, best_score)
    return (False, None, best_score)

def insert_escalation(customer, issue, sentiment, urgency, severity,
                      criticality, category, escalation_flag,
                      likely_to_escalate="No", owner_email="", issue_hash=None,
                      duplicate_of=None):
    """
    Insert a NEW escalation row (no duplicate checks here).
    """
    ensure_schema()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    priority = "high" if str(severity).lower() == "critical" or str(urgency).lower() == "high" else "normal"
    issue_hash = issue_hash or generate_issue_hash(issue)

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO escalations (
                id, customer, issue, issue_hash, sentiment, urgency, severity, criticality, category,
                status, timestamp, action_taken, owner, owner_email, escalated, priority,
                likely_to_escalate, action_owner, status_update_date, user_feedback, duplicate_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            new_id, customer, issue, issue_hash, sentiment, urgency, severity, criticality, category,
            "Open", now, "", "", owner_email or "", escalation_flag, priority,
            likely_to_escalate, "", "", "", duplicate_of
        ))
        conn.commit()
    except Exception as e:
        st.error(f"DB insert failed for {new_id}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return new_id

def add_or_skip_escalation(customer, issue,
                           sentiment, urgency, severity, criticality, category,
                           escalation_flag, likely_to_escalate="No", owner_email=""):
    """
    De-duplicating wrapper:
      - If duplicate found: skip insertion, log audit, return (False, duplicate_id, score)
      - Else: insert and return (True, new_id, 1.0)
    """
    h = generate_issue_hash(issue)
    # If we've processed this exact normalized text before (email ingestion), skip early
    if _processed_hash_exists(h):
        return (False, None, 1.0)

    is_dup, dup_id, score = find_duplicate(issue, customer=customer)
    if is_dup and dup_id:
        try:
            log_escalation_action("duplicate_detected", dup_id, "system",
                                  f"duplicate skipped; score={score:.3f}; customer={customer}")
        except Exception:
            pass
        _mark_processed_hash(h)
        return (False, dup_id, score)

    # Not a duplicate — insert
    new_id = insert_escalation(customer, issue, sentiment, urgency, severity, criticality,
                               category, escalation_flag, likely_to_escalate, owner_email,
                               issue_hash=h, duplicate_of=None)
    _mark_processed_hash(h)
    return (True, new_id, 1.0)

def update_escalation_status(esc_id, status, action_taken, action_owner,
                             owner_email=None, feedback=None, sentiment=None,
                             criticality=None, notes=None):
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
    We do NOT insert here; insertion happens via add_or_skip_escalation so that dedup is applied.
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
                    summary = summarize_issue_text(full_text)

                    # Use both: summary for UI, and normalized hash for ledger
                    hash_val = generate_issue_hash(full_text)

                    # Guard against repeated fetch within same session
                    if hash_val in global_seen_hashes:
                        continue
                    global_seen_hashes.add(hash_val)

                    emails_out.append({"customer": from_, "issue": summary, "raw_hash": hash_val})
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
    """Rules-based tagging for sentiment, urgency, category, severity, criticality, escalation flag."""
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
    """Train small RF model when enough data exists; else return None (falls back to rules)."""
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
    """Yes/No prediction using model; else 2-of-3 risk fallback."""
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
    Background loop: checks inbox every 60s, analyzes new emails, de-dups, inserts.
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
                created, new_or_dup_id, score = add_or_skip_escalation(
                    customer, issue, sent, urg, sev, crit, cat, esc, likely_to_escalate
                )
                # (Optional) You can show a toast in Streamlit logs:
                # st.toast(f"{'New' if created else 'Dup'}: {new_or_dup_id} (score={score:.2f})")
        time.sleep(60)

# ================
# Streamlit Setup
# ================
st.set_page_config(page_title="Escalation Management", layout="wide")

ensure_schema()
try:
    validate_escalation_schema()
except Exception:
    pass
try:
    load_custom_plugins()
except Exception:
    pass

# Header
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
    "📈 Advanced Analytics",   # ← renamed
    "🔥 SLA Heatmap",
    "🧠 Enhancements",
    "⚙️ Admin Tools"
])

# Sidebar: Email Integration
st.sidebar.markdown("### 📩 Email Integration")
if st.sidebar.button("Fetch Emails"):
    emails = parse_emails()
    model = train_model()
    for e in emails:
        issue, customer = e["issue"], e["customer"]
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        likely_to_escalate = predict_escalation(model, sentiment, urgency, severity, criticality)
        created, new_or_dup_id, score = add_or_skip_escalation(
            customer, issue, sentiment, urgency, severity, criticality, category,
            escalation_flag, likely_to_escalate
        )
    st.sidebar.success(f"✅ Processed {len(emails)} unread email(s). De-dup applied.")

# Sidebar: Upload & Analyze
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
        processed_count, dup_count = 0, 0
        for idx, row in df_excel.iterrows():
            issue = str(row.get("Issue", "")).strip()
            customer = str(row.get("Customer", "Unknown")).strip()
            if not issue:
                st.warning(f"⚠️ Row {idx + 1} skipped: empty issue text.")
                continue
            issue_summary = summarize_issue_text(issue)
            sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue_summary)
            likely_to_escalate = predict_escalation(model, sentiment, urgency, severity, criticality)
            created, new_or_dup_id, score = add_or_skip_escalation(
                customer, issue_summary, sentiment, urgency, severity, criticality, category,
                escalation_flag, likely_to_escalate
            )
            if created:
                processed_count += 1
            else:
                dup_count += 1
        st.sidebar.success(f"🎯 {processed_count} new row(s) inserted. 🧹 {dup_count} duplicate(s) skipped.")

# Sidebar: SLA Monitor
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

# Sidebar: Escalation Filters
st.sidebar.markdown("### 🔍 Escalation Filters")
status_opt    = st.sidebar.selectbox("Status",   ["All", "Open", "In Progress", "Resolved"], index=0)
severity_opt  = st.sidebar.selectbox("Severity", ["All", "minor", "major", "critical"], index=0)
sentiment_opt = st.sidebar.selectbox("Sentiment",["All", "positive", "neutral", "negative"], index=0)
category_opt  = st.sidebar.selectbox("Category", ["All", "technical", "support", "dissatisfaction", "safety", "business", "other"], index=0)

# Sidebar: Manual Alerts
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

# Sidebar: WhatsApp Alerts
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

# Sidebar: Downloads
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

# Sidebar: Behavior toggles
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 10, 60, 30)
compact_mode = st.sidebar.checkbox("📱 Compact Mode", value=False)
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
if st.sidebar.button("🔁 Manual Refresh"):
    st.rerun()

# Sidebar: Dark Mode / Summary / PDF
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
    """Simple full-text search across common fields."""
    if not query:
        return df
    q = str(query).strip().lower()
    if df.empty:
        return df
    cols = ['id','customer','issue','owner','action_owner','owner_email',
            'category','severity','sentiment','status']
    present = [c for c in cols if c in df.columns]
    combined = df[present].astype(str).apply(lambda s: s.str.lower()).agg(' '.join, axis=1)
    return df[combined.str.contains(q, na=False, regex=False)]

# ===============================
# Main Page Routing
# ===============================
if page == "📊 Main Dashboard":
    df_all = fetch_escalations()
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')

    filtered_df = df_all.copy()
    if status_opt != "All":
        filtered_df = filtered_df[filtered_df["status"].str.strip().str.title() == status_opt]
    if severity_opt != "All":
        filtered_df = filtered_df[filtered_df["severity"].str.lower() == severity_opt.lower()]
    if sentiment_opt != "All":
        filtered_df = filtered_df[filtered_df["sentiment"].str.lower() == sentiment_opt.lower()]
    if category_opt != "All":
        filtered_df = filtered_df[filtered_df["category"].str.lower() == category_opt.lower()]

    breaches_banner = filtered_df[(filtered_df['status'].str.title() != 'Resolved') &
                                  (filtered_df['priority'].str.lower() == 'high') &
                                  ((datetime.datetime.now() - filtered_df['timestamp']) > datetime.timedelta(minutes=10))]
    if not breaches_banner.empty:
        st.markdown(
            f"<div style='background:#dc3545;padding:8px;border-radius:5px;color:white;text-align:center;'>"
            f"<strong>🚨 {len(breaches_banner)} SLA Breach(s) Detected</strong></div>",
            unsafe_allow_html=True
        )

    # Tabs — Help moved to LAST; Analytics renamed to Summary Analytics
    tabs = st.tabs(["🗃️ All", "🚩 Likely to Escalate", "🔁 Feedback & Retraining", "📊 Summary Analytics", "ℹ️ How this Dashboard Works"])

    # --------------------- Tab 0: All ---------------------
    with tabs[0]:
        st.subheader("📊 Escalation Kanban Board — All Cases")
        search_all = st.text_input("🔍 Search cases", placeholder="Search by ID, customer, issue, owner, email, status…")
        df_view = filter_df_by_query(filtered_df.copy(), search_all)
        df_view["status"] = df_view["status"].fillna("Open").str.strip().str.title()
        counts = df_view['status'].value_counts()
        col1, col2, col3 = st.columns(3)
        status_columns = {"Open": col1, "In Progress": col2, "Resolved": col3}

        # Train model once per render
        model_for_view = train_model()

        for status_name, col in status_columns.items():
            with col:
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
                        sentiment = (row.get("sentiment") or "neutral").lower()
                        urgency = (row.get("urgency") or "normal").lower()
                        severity = (row.get("severity") or "minor").lower()
                        criticality = (row.get("criticality") or "medium").lower()
                        likely_to_escalate = predict_escalation(model_for_view, sentiment, urgency, severity, criticality)
                        flag = "🚩" if likely_to_escalate == 'Yes' else ""
                        expander_label = f"{row.get('id', 'N/A')} - {row.get('customer', 'Unknown')} {flag} – {summary}"
                        prefix = f"case_{row.get('id', 'N/A')}"
                        header_color = SEVERITY_COLORS.get(severity, "#7f8c8d")
                        urgency_color = URGENCY_COLORS.get(urgency, "#7f8c8d")
                        sentiment_cap = (row.get("sentiment") or "neutral").capitalize()
                        sentiment_color = {"Negative": "#e74c3c", "Positive": "#2ecc71", "Neutral": "#f39c12"}.get(sentiment_cap, "#7f8c8d")
                        escalated_color = "#c0392b" if likely_to_escalate == "Yes" else "#7f8c8d"
                        category = (row.get("category") or "other").capitalize()
                        criticality_cap = (row.get("criticality") or "medium").capitalize()
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
                            if not compact_mode:
                                colA, colB, colC, colD = st.columns([1, 2, 2, 1])
                                with colA:
                                    st.markdown(
                                        f"<div style='background-color:{ageing_color};padding:6px;border-radius:5px;"
                                        f"color:white;text-align:center'>Age: {ageing_str}</div>",
                                        unsafe_allow_html=True
                                    )
                                with colB:
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
                                    n1_email = st.text_input("N+1 Email", key=f"{prefix}_n1email")
                                with colD:
                                    if st.button("🚀 To N+1", key=f"{prefix}_n1btn"):
                                        update_escalation_status(row['id'], row.get("status", "Open"),
                                                                 row.get("action_taken", ""),
                                                                 row.get("owner", ""),
                                                                 n1_email)
                                        if n1_email:
                                            send_alert("Case escalated to N+1.", via="email", recipient=n1_email)
                                        send_alert("Case escalated to N+1.", via="teams")

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

                            edit_row1_col1, edit_row1_col2 = st.columns(2)
                            with edit_row1_col1:
                                current_status = (row.get("status") or "Open").strip().title()
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

    # ----------------- Tab 1: Likely to Escalate -----------------
    with tabs[1]:
        st.subheader("🚩 Likely to Escalate")
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

    # ----------------- Tab 2: Feedback & Retraining -----------------
    with tabs[2]:
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

    # ----------------- Tab 3: Summary Analytics -----------------
    with tabs[3]:
        st.subheader("📊 Summary Analytics")
        try:
            render_analytics()
        except Exception as e:
            st.info("Analytics module not fully configured.")
            st.exception(e)

    # ----------------- Tab 4: Help (LAST) -----------------
    with tabs[4]:
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

elif page == "📈 Advanced Analytics":   # ← updated routing
    try:
        show_analytics_view()
    except Exception as e:
        st.error("❌ Failed to load analytics view.")
        st.exception(e)

elif page == "⚙️ Admin Tools":
    def show_admin_panel():
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
    email_thread = threading.Thread(target=email_polling_job, daemon=True)
    email_thread.start()
    st.session_state['email_thread'] = email_thread

# --------------------------
# Daily Email Scheduler
# --------------------------
def send_daily_escalation_email():
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
