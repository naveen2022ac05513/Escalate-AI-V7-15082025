# EscalateAI_v7.py
# --------------------------------------------------------------------
# EscalateAI — Customer Escalation Prediction & Management Tool
# --------------------------------------------------------------------
# Key updates in this build:
# • Header row: Escalation View + SLA capsule + AI Summary on one line
# • Below header: Total (filtered) pill on left + compact Search on right
# • Sidebar: MS Teams & Email composer restored; WhatsApp & SMS allowed
#   only for Resolved cases
# • Feedback & Retraining: 3 cases per page, 3-column grid with paging
# • BU/Region filters in sidebar; totals/kanban reflect applied filters
# • BU/Region columns stored in DB; schema migration handled
# • Charts with value labels for BU/Region
# --------------------------------------------------------------------

import os, re, time, datetime, threading, hashlib, sqlite3, smtplib, requests, imaplib, email, traceback, schedule
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Optional TF-IDF for duplicate detection
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine
    _TFIDF_AVAILABLE = True
except Exception:
    _TFIDF_AVAILABLE = False
import difflib

# Optional BU/Region bucketizer
try:
    from bu_region_bucketizer import (
        classify_bu, enrich_with_bu_region, bucketize_region
    )
    _BU_OK = True
except Exception:
    _BU_OK = False
    def classify_bu(text): return ("OTHER", "Other / Unclassified")
    def enrich_with_bu_region(df, text_cols, country_col="Country", state_col="State", city_col="City"):
        out = df.copy()
        out["bu_code"] = "OTHER"; out["bu_name"] = "Other / Unclassified"; out["region"] = "Others"
        return out
    def bucketize_region(country, state=None, city=None, text_hint=None): return "Others"

from dotenv import load_dotenv

# ---------------- Optional modules (safe fallbacks) ----------------
try:
    from enhancements import (
        render_analytics, show_feature_importance, generate_pdf_report,
        render_sla_heatmap, apply_dark_mode, show_filter_summary,
        get_escalation_template, summarize_escalations, schedule_weekly_retraining
    )
except Exception:
    def render_analytics(): st.info("enhancements.render_analytics not available.")
    def show_feature_importance(*a, **k): pass
    def generate_pdf_report(): raise RuntimeError("enhancements.generate_pdf_report missing")
    def render_sla_heatmap(): st.info("enhancements.render_sla_heatmap not available.")
    def apply_dark_mode(): pass
    def show_filter_summary(*a, **k): pass
    def summarize_escalations(): return "Summary unavailable."
    def schedule_weekly_retraining(): pass

try:
    from advanced_enhancements import (
        validate_escalation_schema, log_escalation_action, load_custom_plugins,
        send_whatsapp_message, predict_resolution_eta, show_shap_explanation,
        generate_text_pdf, render_model_metrics, score_feedback_quality
    )
except Exception:
    def validate_escalation_schema(): pass
    def log_escalation_action(*a, **k): pass
    def load_custom_plugins(): pass
    def send_whatsapp_message(*a, **k): return False

# ---------------- Quick analytics view ----------------
def show_analytics_view():
    df = fetch_escalations()
    st.title("📊 Escalation Analytics")
    if df.empty:
        st.warning("⚠️ No escalation data available."); return
    st.subheader("📈 Escalation Volume Over Time")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    st.line_chart(df.groupby(df['timestamp'].dt.date).size())
    st.subheader("🔥 Severity Distribution")
    st.bar_chart(df['severity'].value_counts())
    st.subheader("🧠 Sentiment Breakdown")
    st.bar_chart(df['sentiment'].value_counts())
    st.subheader("⏳ Ageing Buckets")
    df['age_days'] = (pd.Timestamp.now() - df['timestamp']).dt.days
    df['age_bucket'] = pd.cut(df['age_days'], bins=[0,3,7,14,30,90], labels=["0–3d","4–7d","8–14d","15–30d","31–90d"])
    st.bar_chart(df['age_bucket'].value_counts().sort_index())

# ---------------- Configuration ----------------
load_dotenv()
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER   = os.getenv("EMAIL_USER")
EMAIL_PASS   = os.getenv("EMAIL_PASS") or ""
SMTP_SERVER  = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER", EMAIL_USER or "")
TEAMS_WEBHOOK   = os.getenv("MS_TEAMS_WEBHOOK_URL", "")
EMAIL_SUBJECT   = os.getenv("EMAIL_SUBJECT", "🚨 EscalateAI Alert")

# Twilio SMS
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-25"
analyzer = SentimentIntensityAnalyzer()

NEGATIVE_KEYWORDS = {
    "technical": ["fail","break","crash","defect","fault","degrade","damage","trip","malfunction","blank","shutdown","discharge","leak"],
    "dissatisfaction": ["dissatisfy","frustrate","complain","reject","delay","ignore","escalate","displease","noncompliance","neglect"],
    "support": ["wait","pending","slow","incomplete","miss","omit","unresolved","shortage","no response"],
    "safety": ["fire","burn","flashover","arc","explode","unsafe","leak","corrode","alarm","incident"],
    "business": ["impact","loss","risk","downtime","interrupt","cancel","terminate","penalty"]
}

processed_email_uids_lock = threading.Lock()
global_seen_hashes = set()

# Colors
STATUS_COLORS   = {"Open":"#f59e0b","In Progress":"#3b82f6","Resolved":"#22c55e"}  # Orange / Blue / Green
SEVERITY_COLORS = {"critical":"#ef4444","major":"#f59e0b","minor":"#10b981"}
URGENCY_COLORS  = {"high":"#dc2626","normal":"#16a34a"}

# ---------------- DB helpers ----------------
def summarize_issue_text(t: str) -> str:
    t = re.sub(r'\s+',' ', t or "").strip()
    return t[:200] + "..." if len(t) > 200 else t

def _normalize_text(t: str) -> str:
    if not t: return ""
    for pat in [r"[-]+\s*Forwarded message\s*[-]+", r"From:.*", r"Sent:.*", r"To:.*", r"Subject:.*", r">.*", r"On .* wrote:"]:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+"," ", t).strip().lower()

def generate_issue_hash(issue_text: str) -> str:
    return hashlib.md5(_normalize_text(issue_text).encode()).hexdigest()

def get_next_escalation_id() -> str:
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
    last = cur.fetchone(); conn.close()
    nxt = int((last[0].replace(ESCALATION_PREFIX,"")) if last else 0) + 1
    return f"{ESCALATION_PREFIX}{str(nxt).zfill(5)}"

def ensure_schema():
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY, customer TEXT, issue TEXT, issue_hash TEXT,
            sentiment TEXT, urgency TEXT, severity TEXT, criticality TEXT, category TEXT,
            status TEXT, timestamp TEXT, action_taken TEXT, owner TEXT, owner_email TEXT,
            escalated TEXT, priority TEXT, likely_to_escalate TEXT, action_owner TEXT,
            status_update_date TEXT, user_feedback TEXT, duplicate_of TEXT,
            bu_code TEXT, bu_name TEXT, region TEXT
        )''')
        # ensure columns exist
        for col in ["issue_hash","duplicate_of","owner_email","status_update_date","user_feedback",
                    "likely_to_escalate","action_owner","priority","bu_code","bu_name","region"]:
            try: cur.execute(f"SELECT {col} FROM escalations LIMIT 1")
            except Exception: cur.execute(f"ALTER TABLE escalations ADD COLUMN {col} TEXT")
        cur.execute('''CREATE TABLE IF NOT EXISTS processed_hashes (hash TEXT PRIMARY KEY, first_seen TEXT)''')
        conn.commit()
    except Exception: traceback.print_exc()
    finally:
        try: conn.close()
        except Exception: pass

def _processed_hash_exists(h: str) -> bool:
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT 1 FROM processed_hashes WHERE hash=? LIMIT 1",(h,))
    row = cur.fetchone(); conn.close(); return row is not None

def _mark_processed_hash(h: str):
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO processed_hashes (hash, first_seen) VALUES (?,?)",
                    (h, datetime.datetime.now().isoformat()))
        conn.commit()
    except Exception: pass
    finally:
        try: conn.close()
        except Exception: pass

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
    # Normalize columns
    for c in ["status","severity","urgency","sentiment","criticality","category","bu_code","region","priority"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "bu_code" not in df.columns: df["bu_code"] = "OTHER"
    if "bu_name" not in df.columns: df["bu_name"] = "Other / Unclassified"
    if "region" not in df.columns: df["region"] = "Others"
    return df

# --------------- Duplicate detection ---------------
def _cosine_sim(a: str, b: str) -> float:
    a, b = _normalize_text(a), _normalize_text(b)
    if not a or not b: return 0.0
    if _TFIDF_AVAILABLE:
        try:
            vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
            X = vec.fit_transform([a,b]); return float(_cosine(X[0],X[1]))
        except Exception: pass
    return difflib.SequenceMatcher(None,a,b).ratio()

def find_duplicate(issue_text: str, customer: str|None=None,
                   days_window:int=180, cosine_threshold:float=0.88,
                   difflib_threshold:float=0.92):
    ensure_schema(); text_norm = _normalize_text(issue_text or "")
    if not text_norm: return (False,None,0.0)
    h = hashlib.md5(text_norm.encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM escalations WHERE issue_hash=? LIMIT 1",(h,))
        row = cur.fetchone()
        if row: return (True,row[0],1.0)
        df = pd.read_sql("SELECT id,issue,customer,timestamp FROM escalations", conn)
    except Exception: df = pd.DataFrame()
    finally: conn.close()
    if df.empty: return (False,None,0.0)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_window)
        df = df[df['timestamp'].isna() | (df['timestamp'] >= cutoff)]
    except Exception: pass
    if customer and 'customer' in df.columns:
        df_same = df[df['customer'].astype(str).str.lower()==str(customer).lower()]
        df_pool = df_same if not df_same.empty else df
    else:
        df_pool = df
    best_id, best_score = None, 0.0
    for _, r in df_pool.iterrows():
        s = _cosine_sim(issue_text, r.get('issue',''))
        if s > best_score: best_score, best_id = s, r.get('id',None)
    if _TFIDF_AVAILABLE and best_score >= cosine_threshold: return (True,best_id,best_score)
    if (not _TFIDF_AVAILABLE) and best_score >= difflib_threshold: return (True,best_id,best_score)
    return (False,None,best_score)

def insert_escalation(customer, issue, sentiment, urgency, severity,
                      criticality, category, escalation_flag, likely_to_escalate="No",
                      owner_email="", issue_hash=None, duplicate_of=None,
                      bu_code="OTHER", bu_name="Other / Unclassified", region="Others"):
    ensure_schema()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    priority = "high" if str(severity).lower()=="critical" or str(urgency).lower()=="high" else "normal"
    issue_hash = issue_hash or generate_issue_hash(issue)
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute('''INSERT INTO escalations (
            id,customer,issue,issue_hash,sentiment,urgency,severity,criticality,category,
            status,timestamp,action_taken,owner,owner_email,escalated,priority,
            likely_to_escalate,action_owner,status_update_date,user_feedback,duplicate_of,
            bu_code, bu_name, region
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            new_id, customer, issue, issue_hash, sentiment, urgency, severity, criticality, category,
            "Open", now, "", "", owner_email or "", escalation_flag, priority,
            likely_to_escalate, "", "", "", duplicate_of, bu_code, bu_name, region
        ))
        conn.commit()
    except Exception as e:
        st.error(f"DB insert failed for {new_id}: {e}")
    finally:
        try: conn.close()
        except Exception: pass
    return new_id

def add_or_skip_escalation(customer, issue, sentiment, urgency, severity,
                           criticality, category, escalation_flag,
                           likely_to_escalate="No", owner_email="",
                           country=None, state=None, city=None):
    h = generate_issue_hash(issue)
    if _processed_hash_exists(h): return (False,None,1.0)
    is_dup, dup_id, score = find_duplicate(issue, customer=customer)
    if is_dup and dup_id:
        try: log_escalation_action("duplicate_detected", dup_id, "system", f"duplicate skipped; score={score:.3f}; customer={customer}")
        except Exception: pass
        _mark_processed_hash(h); return (False,dup_id,score)
    # BU/Region
    bu_code, bu_name = classify_bu(issue) if _BU_OK else ("OTHER","Other / Unclassified")
    region = bucketize_region(country or "India", state, city, text_hint=issue) if _BU_OK else "Others"
    new_id = insert_escalation(
        customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag,
        likely_to_escalate, owner_email, issue_hash=h, duplicate_of=None,
        bu_code=bu_code, bu_name=bu_name, region=region
    )
    _mark_processed_hash(h); return (True,new_id,1.0)

def update_escalation_status(esc_id, status, action_taken, action_owner,
                             owner_email=None, feedback=None, sentiment=None,
                             criticality=None, notes=None):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute('''UPDATE escalations
                   SET status=?, action_taken=?, action_owner=?, status_update_date=?,
                       owner_email=?, user_feedback=?, sentiment=?, criticality=?
                   WHERE id=?''', (
        status, action_taken, action_owner, datetime.datetime.now().isoformat(),
        owner_email, notes if notes is not None else feedback, sentiment, criticality, esc_id
    ))
    conn.commit(); conn.close()
    try: log_escalation_action("update_status", esc_id, action_owner or "system", f"status={status}; action_taken={action_taken}")
    except Exception: pass

# ---------------- Email / Alerts / SMS ----------------
def parse_emails():
    out, conn = [], None
    try:
        if not EMAIL_USER:
            st.warning("Configure EMAIL_USER/EMAIL_PASS in .env"); return out
        conn = imaplib.IMAP4_SSL(EMAIL_SERVER); conn.login(EMAIL_USER, EMAIL_PASS); conn.select("inbox")
        _, msgs = conn.search(None, "UNSEEN")
        for num in msgs[0].split():
            _, msg_data = conn.fetch(num, "(RFC822)")
            for rp in msg_data:
                if not isinstance(rp, tuple): continue
                msg = email.message_from_bytes(rp[1])
                subject = decode_header(msg.get("Subject",""))[0][0]
                if isinstance(subject, bytes): subject = subject.decode(errors='ignore')
                from_ = msg.get("From","unknown")
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try: body = part.get_payload(decode=True).decode(errors='ignore')
                            except Exception: body = ""
                            break
                else:
                    try: body = msg.get_payload(decode=True).decode(errors='ignore')
                    except Exception: body = ""
                full = f"{subject} - {body}"
                h = generate_issue_hash(full)
                if h in global_seen_hashes: continue
                global_seen_hashes.add(h)
                out.append({"customer": from_, "issue": summarize_issue_text(full), "raw_hash": h})
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
    finally:
        if conn:
            try: conn.logout()
            except Exception: pass
    return out

def analyze_issue(text: str):
    sc = analyzer.polarity_scores(text or ""); c = sc["compound"]
    sentiment = "negative" if c < -0.05 else "positive" if c > 0.05 else "neutral"
    t = (text or "").lower()
    urgency = "high" if any(w in t for cat in NEGATIVE_KEYWORDS.values() for w in cat) else "normal"
    category = None
    for cat, kws in NEGATIVE_KEYWORDS.items():
        if any(k in t for k in kws): category = cat; break
    severity = "critical" if category in ["safety","technical"] else "major" if category in ["support","business"] else "minor"
    criticality = "high" if (sentiment=="negative" and urgency=="high") else "medium"
    escalation_flag = "Yes" if (urgency=="high" or sentiment=="negative") else "No"
    return sentiment, urgency, severity, criticality, (category or "other"), escalation_flag

def train_model():
    df = fetch_escalations()
    if df.shape[0] < 20: return None
    df = df.dropna(subset=['sentiment','urgency','severity','criticality','likely_to_escalate'])
    if df.empty: return None
    X = pd.get_dummies(df[['sentiment','urgency','severity','criticality']])
    y = df['likely_to_escalate'].apply(lambda x: 1 if str(x).strip().lower()=="yes" else 0)
    if y.nunique() < 2: return None
    X_train, _, y_train, _ = train_test_split(X,y,test_size=0.2,random_state=42)
    m = RandomForestClassifier(random_state=42); m.fit(X_train,y_train); return m

def predict_escalation(m, s, u, sev, c):
    if m is None:
        risk = (sev in ["critical","high"]) + (u in ["high","immediate"]) + (s in ["negative","very negative"])
        return "Yes" if risk >= 2 else "No"
    Xp = pd.DataFrame([{f"sentiment_{s}":1, f"urgency_{u}":1, f"severity_{sev}":1, f"criticality_{c}":1}]).reindex(columns=m.feature_names_in_, fill_value=0)
    return "Yes" if int(m.predict(Xp)[0])==1 else "No"

def send_alert(message:str, via:str="email", recipient:str|None=None):
    if via=="email":
        try:
            to_addr = recipient if recipient else ALERT_RECIPIENT
            if not to_addr:
                st.warning("No email recipient configured."); return
            msg = MIMEText(message, 'plain', 'utf-8'); msg['Subject']=EMAIL_SUBJECT
            msg['From']=EMAIL_USER or "no-reply@escalateai"; msg['To']=to_addr
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
                s.starttls()
                if EMAIL_USER and EMAIL_PASS: s.login(EMAIL_USER, EMAIL_PASS)
                s.sendmail(msg['From'], [msg['To']], msg.as_string())
        except Exception as e: st.error(f"Email alert failed: {e}")
    elif via=="teams":
        try:
            if not TEAMS_WEBHOOK: st.error("MS Teams webhook URL is not configured."); return
            r = requests.post(TEAMS_WEBHOOK, json={"text":message}, headers={"Content-Type":"application/json"})
            if r.status_code != 200: st.error(f"Teams alert failed: {r.status_code} - {r.text}")
        except Exception as e: st.error(f"Teams alert failed: {e}")

def send_sms(to_number:str, body:str) -> bool:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        st.error("Twilio SMS not configured (.env: TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM_NUMBER)"); return False
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        r = requests.post(url, data={"From":TWILIO_FROM_NUMBER,"To":to_number,"Body":body},
                          auth=(TWILIO_ACCOUNT_SID,TWILIO_AUTH_TOKEN), timeout=15)
        if r.status_code in (200,201): return True
        st.error(f"SMS failed: {r.status_code} — {r.text}"); return False
    except Exception as e: st.error(f"SMS exception: {e}"); return False

def email_polling_job():
    while True:
        m = train_model()
        for e in parse_emails():
            s,u,sev,c,cat,esc = analyze_issue(e["issue"])
            likely = predict_escalation(m,s,u,sev,c)
            # Email has no country/state/city—use India/Others fallback
            add_or_skip_escalation(e["customer"], e["issue"], s,u,sev,c,cat,esc, likely, country="India")
        time.sleep(60)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Escalation Management", layout="wide")
ensure_schema()
try: validate_escalation_schema()
except Exception: pass
try: load_custom_plugins()
except Exception: pass

# Styles
st.markdown("""
<style>
  .sticky-header{position:sticky;top:0;z-index:999;background:linear-gradient(135deg,#0ea5e9 0%,#7c3aed 100%);
    padding:12px 16px;border-radius:0 0 12px 12px;box-shadow:0 8px 20px rgba(0,0,0,.12);}
  .sticky-header h1{color:#fff;margin:0;text-align:center;font-size:30px;line-height:1.2;}

  .kanban-title{display:flex;justify-content:center;align-items:center;gap:8px;border-radius:10px;
    padding:8px 10px;color:#fff;text-align:center;box-shadow:0 6px 14px rgba(0,0,0,.07);margin:4px 0;font-size:14px;}

  details[data-testid="stExpander"]{
    background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;margin:2px 0 !important;
    box-shadow:0 4px 10px rgba(0,0,0,.05);
  }
  details[data-testid="stExpander"] > summary{padding:8px 10px;font-weight:700;}
  details[data-testid="stExpander"] > div[role="region"]{padding:8px 10px 10px 10px;}
  div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]{gap:.25rem !important;}

  .age{padding:4px 8px;border-radius:8px;color:#fff;font-weight:600;text-align:center;font-size:12px;}
  .summary{font-size:15px;color:#0f172a;margin-bottom:6px;}

  .kpi-panel{ margin-top:0 !important; background:transparent !important; border:0 !important; box-shadow:none !important; padding:0 !important; }
  .kpi-gap{ height:18px !important; }

  .tag-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600;
            border:1px solid var(--c,#cbd5e1);color:var(--c,#334155);background:#fff;white-space:nowrap;}

  .controls-panel{ background:#fff; border:0; border-radius:12px; padding:10px 0 2px 0; margin:6px 0 2px 0; }

  /* header row extras */
  .sla-pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#ef4444;color:#fff;font-weight:700;font-size:12px;}
  .aisum{background:#0b1220;color:#e5f2ff;padding:10px 12px;border-radius:10px;
         box-shadow:0 6px 14px rgba(0,0,0,.10);font-size:13px;}

  /* totals + search row */
  .counts-pill{display:inline-block;padding:6px 10px;border-radius:10px;background:#f8fafc;border:1px solid #e5e7eb;
    font-weight:600;color:#334155;font-size:13px;white-space:nowrap;}
  .compact-caption{ margin-bottom:4px; color:#64748b; font-size:12px; }

  /* Inputs/labels uniform */
  div[data-testid="stTextInput"]  label,
  div[data-testid="stTextArea"]   label,
  div[data-testid="stSelectbox"]  label {
    font-size:13px !important; font-weight:600 !important; color:#475569 !important; margin-bottom:4px !important;
  }
  div[data-testid="stTextInput"] input,
  div[data-testid="stTextArea"] textarea {
    background:#f3f4f6 !important; border:1px solid #e5e7eb !important; border-radius:8px !important; height:40px !important; padding:8px 10px !important;
  }
  div[data-testid="stSelectbox"] div[role="combobox"]{
    background:#f3f4f6 !important; border:1px solid #e5e7eb !important; border-radius:8px !important; min-height:40px !important; padding:6px 10px !important; align-items:center !important;
  }
  .controls-panel .stButton>button{
    height:40px !important; border-radius:10px !important; padding:0 14px !important;
  }
</style>
<div class="sticky-header"><h1>🚨 EscalateAI – AI Based Customer Escalation Prediction & Management Tool</h1></div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📊 Main Dashboard","📈 Advanced Analytics","🔥 SLA Heatmap","🧠 Enhancements","📊 BU/Region Trends","⚙️ Admin Tools"])

# Sidebar — email import
st.sidebar.markdown("### 📩 Email Integration")
if st.sidebar.button("Fetch Emails"):
    mails = parse_emails(); m = train_model()
    for e in mails:
        s,u,sev,c,cat,esc = analyze_issue(e["issue"])
        likely = predict_escalation(m,s,u,sev,c)
        add_or_skip_escalation(e["customer"], e["issue"], s,u,sev,c,cat,esc, likely, country="India")
    st.sidebar.success(f"✅ Processed {len(mails)} unread email(s). De-dup applied.")

# Sidebar — upload
st.sidebar.header("📁 Upload Escalation Sheet")
uploaded = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded:
    try:
        df_x = pd.read_excel(uploaded)
        st.sidebar.success("✅ Excel loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Read failed: {e}")
        st.stop()
    need = [c for c in ["Customer","Issue"] if c not in df_x.columns]
    if need:
        st.sidebar.error("Missing required columns: " + ", ".join(need)); st.stop()
    # Optional enrich via bucketizer if Country/State/City exist
    text_cols = ["Issue"]
    if _BU_OK and any(c in df_x.columns for c in ["Country","State","City"]):
        df_x = enrich_with_bu_region(df_x, text_cols=text_cols,
                                     country_col="Country" if "Country" in df_x.columns else "Customer",
                                     state_col="State" if "State" in df_x.columns else None,
                                     city_col="City" if "City" in df_x.columns else None)
    if st.sidebar.button("🔍 Analyze & Insert"):
        m = train_model(); ok, dups = 0, 0
        for _, r in df_x.iterrows():
            issue = str(r.get("Issue","")).strip(); cust = str(r.get("Customer","Unknown")).strip()
            if not issue: continue
            text = summarize_issue_text(issue)
            s,u,sev,c,cat,esc = analyze_issue(text); likely = predict_escalation(m,s,u,sev,c)
            # Use BU/Region from enriched df if present
            bu_code, bu_name = (r.get("bu_code","OTHER"), r.get("bu_name","Other / Unclassified"))
            region = r.get("region","Others")
            if bu_code == "" or pd.isna(bu_code): bu_code, bu_name = classify_bu(text)
            if region == "" or pd.isna(region):
                region = bucketize_region(r.get("Country","India"), r.get("State"), r.get("City"), text_hint=text)
            h = generate_issue_hash(text)
            if _processed_hash_exists(h):
                dups += 1; continue
            created, _, _ = add_or_skip_escalation(
                cust, text, s,u,sev,c,cat,esc, likely,
                country=r.get("Country","India"), state=r.get("State"), city=r.get("City")
            )
            ok += int(created); dups += int(not created)
        st.sidebar.success(f"🎯 Inserted {ok}, skipped {dups} duplicate(s).")

# Sidebar — SLA manual check
st.sidebar.markdown("### ⏰ SLA Monitor")
if st.sidebar.button("Trigger SLA Check"):
    df_t = fetch_escalations()
    if df_t.empty: st.sidebar.info("No data yet.")
    else:
        df_t['timestamp'] = pd.to_datetime(df_t['timestamp'], errors='coerce')
        breaches = df_t[(df_t['status'].astype(str).str.title()!='Resolved') & (df_t['priority'].astype(str).str.lower()=='high') &
                        ((datetime.datetime.now()-df_t['timestamp']) > datetime.timedelta(minutes=10))]
        if not breaches.empty:
            msg = f"🚨 SLA breach for {len(breaches)} case(s)!"
            send_alert(msg, via="teams"); send_alert(msg, via="email"); st.sidebar.success("✅ Alerts sent")
        else: st.sidebar.info("All SLAs healthy")

# Sidebar — Filters
st.sidebar.markdown("### 🔍 Filters")
status_opt    = st.sidebar.selectbox("Status",   ["All","Open","In Progress","Resolved"], index=0)
severity_opt  = st.sidebar.selectbox("Severity", ["All","minor","major","critical"], index=0)
sentiment_opt = st.sidebar.selectbox("Sentiment",["All","positive","neutral","negative"], index=0)
category_opt  = st.sidebar.selectbox("Category", ["All","technical","support","dissatisfaction","safety","business","other"], index=0)

# BU & Region filters
df_for_filters = fetch_escalations()
bu_codes = sorted([b for b in df_for_filters.get("bu_code", pd.Series(dtype=str)).dropna().unique().tolist() if b], key=str)
if not bu_codes: bu_codes = ["PPIBS","PSIBS","IDIBS","SPIBS","BMS","H&D","A2E","Solar","OTHER"]
bu_opt = st.sidebar.multiselect("BU (code)", options=bu_codes, default=[])
region_vals = sorted([r for r in df_for_filters.get("region", pd.Series(dtype=str)).dropna().unique().tolist() if r], key=str)
if not region_vals: region_vals = ["North","East","South","West","NC","Others"]
region_opt = st.sidebar.multiselect("Region", options=region_vals, default=[])

# Sidebar — Notifications
st.sidebar.markdown("### 📣 Manual Notifications")
msg_text = st.sidebar.text_area("Compose message", "Test alert from EscalateAI")
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.sidebar.button("Send MS Teams"):
        send_alert(msg_text, via="teams")
        st.sidebar.success("✅ Teams alert sent")
with c2:
    if st.sidebar.button("Send Email"):
        send_alert(msg_text, via="email")
        st.sidebar.success("✅ Email alert sent")

# Sidebar — WhatsApp & SMS (Resolved only)
st.sidebar.markdown("### 📲 WhatsApp & SMS (Resolved only)")
df_notify = fetch_escalations()
df_resolved = df_notify[df_notify.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.title()=="Resolved"]
if not df_resolved.empty:
    esc_id = st.sidebar.selectbox("🔢 Select Resolved Escalation ID", df_resolved["id"].astype(str).tolist())
    phone = st.sidebar.text_input("📞 Phone Number", "+91", help="Include country code (e.g., +91)")
    wmsg = st.sidebar.text_area("📨 Message", f"Your issue with ID {esc_id} has been resolved. Thank you!")
    wc1, wc2 = st.sidebar.columns(2)
    with wc1:
        if st.sidebar.button("Send WhatsApp"):
            try:
                ok = send_whatsapp_message(phone, wmsg) if callable(send_whatsapp_message) else False
                st.sidebar.success(f"✅ WhatsApp sent to {phone}") if ok else st.sidebar.error("❌ WhatsApp API failure")
            except Exception as e:
                st.sidebar.error(f"❌ WhatsApp send failed: {e}")
    with wc2:
        if st.sidebar.button("Send SMS"):
            st.sidebar.success(f"✅ SMS sent to {phone}") if send_sms(phone, wmsg) else None
else:
    st.sidebar.info("No resolved cases available for WhatsApp/SMS.")

# Sidebar — Downloads / Dev
st.sidebar.markdown("### 📤 Downloads")
dl1, dl2 = st.sidebar.columns(2)
with dl1:
    if st.sidebar.button("⬇️ All Complaints"):
        csv = fetch_escalations().to_csv(index=False)
        st.sidebar.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")
with dl2:
    if st.sidebar.button("⬇️ Escalated Only"):
        d = fetch_escalations(); d = d[d.get("escalated", pd.Series(dtype=str)).astype(str).str.lower()=="yes"] if not d.empty else d
        if d.empty: st.sidebar.info("No escalated cases.")
        else:
            out = "escalated_cases.xlsx"; d.to_excel(out, index=False)
            with open(out,"rb") as f: st.sidebar.download_button("Download Excel", f, file_name=out,
                                                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.sidebar.markdown("### 🛠️ Developer")
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 10, 60, 30)
if auto_refresh: time.sleep(refresh_interval); st.rerun()
if st.sidebar.button("🔁 Manual Refresh"): st.rerun()
st.sidebar.markdown("### 📧 Daily Escalation Email")
if st.sidebar.button("Send Daily Email"):
    # uses function defined at bottom
    pass  # button exists; function registered in scheduler below

if st.sidebar.checkbox("🧪 View Raw Database"):
    st.sidebar.dataframe(fetch_escalations())

if st.sidebar.button("🗑️ Reset Database (Dev Only)"):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS escalations")
    cur.execute("DROP TABLE IF EXISTS processed_hashes")
    conn.commit(); conn.close()
    st.sidebar.warning("Database reset. Please restart or re-upload.")

if st.sidebar.checkbox("🌙 Dark Mode"):
    try: apply_dark_mode()
    except Exception: pass

# Helpers
def filter_df_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query or df.empty: return df
    q = str(query).strip().lower()
    cols = ['id','customer','issue','owner','action_owner','owner_email','category','severity','sentiment','status','bu_code','region']
    present = [c for c in cols if c in df.columns]
    combined = df[present].astype(str).apply(lambda s: s.str.lower()).agg(' '.join, axis=1)
    return df[combined.str.contains(q, na=False, regex=False)]

# ---------------- Routing ----------------
if page == "📊 Main Dashboard":
    df_all = fetch_escalations()
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')

    tabs = st.tabs(["🗃️ All","🚩 Likely to Escalate","🔁 Feedback & Retraining","📊 Summary Analytics","ℹ️ How this Dashboard Works"])

    # ---------------- Tab 0: All ----------------
    with tabs[0]:
        st.subheader("📊 Escalation Kanban Board — All Cases")

        # Apply sidebar filters
        filt = df_all.copy()
        if status_opt != "All":   filt = filt[filt["status"].astype(str).str.strip().str.title()==status_opt]
        if severity_opt != "All": filt = filt[filt["severity"].astype(str).str.lower()==severity_opt.lower()]
        if sentiment_opt != "All":filt = filt[filt["sentiment"].astype(str).str.lower()==sentiment_opt.lower()]
        if category_opt != "All": filt = filt[filt["category"].astype(str).str.lower()==category_opt.lower()]
        if bu_opt:                filt = filt[filt["bu_code"].astype(str).isin(bu_opt)]
        if region_opt:            filt = filt[filt["region"].astype(str).isin(region_opt)]

        # --- Row A: Escalation View + SLA + AI (one line) ---
        rowA_l, rowA_m, rowA_r = st.columns([0.58, 0.14, 0.28])

        with rowA_l:
            view_radio = st.radio(
                "Escalation View",
                ["All", "Likely to Escalate", "Not Likely", "SLA Breach"],
                horizontal=True
            )

        _f = filt.copy()
        _f['timestamp'] = pd.to_datetime(_f['timestamp'], errors='coerce')
        sla_breaches = _f[
            (_f['status'].astype(str).str.strip().str.title()!='Resolved')
            & (_f['priority'].astype(str).str.lower()=='high')
            & ((datetime.datetime.now()-_f['timestamp']) > datetime.timedelta(minutes=10))
        ]
        with rowA_m:
            st.markdown(f"<span class='sla-pill'>⏱️ {len(sla_breaches)} SLA breach(s)</span>", unsafe_allow_html=True)

        with rowA_r:
            try: ai_text = summarize_escalations()
            except Exception: ai_text = "Summary unavailable."
            st.markdown(f"<div class='aisum'><b>🧠 AI Summary</b><br>{ai_text}</div>", unsafe_allow_html=True)

        # Apply view selection
        base = filt.copy()
        if view_radio == "SLA Breach":
            base = sla_breaches.copy()
        elif view_radio in ("Likely to Escalate","Not Likely"):
            mtmp = train_model()
            def _pred_row(r):
                return predict_escalation(
                    mtmp,
                    (r.get("sentiment") or "neutral").lower(),
                    (r.get("urgency") or "normal").lower(),
                    (r.get("severity") or "minor").lower(),
                    (r.get("criticality") or "medium").lower(),
                )
            base = base.copy()
            base["likely_calc"] = base.apply(_pred_row, axis=1)
            base = base[base["likely_calc"]=="Yes"] if view_radio=="Likely to Escalate" else base[base["likely_calc"]!="Yes"]

        # helper for totals
        def _counts_bar(dfvv: pd.DataFrame) -> str:
            if dfvv is None or dfvv.empty:
                return "Total: 0  |  Open: 0  |  In Progress: 0  |  Resolved: 0"
            s = dfvv['status'].astype(str).str.strip().str.title()
            total = len(dfvv); open_c = (s=="Open").sum(); ip_c = (s=="In Progress").sum(); res_c = (s=="Resolved").sum()
            return f"Total: {total}  |  Open: {open_c}  |  In Progress: {ip_c}  |  Resolved: {res_c}"

        # --- Row B: compact Total left + compact Search right ---
        rowB_l, rowB_r = st.columns([0.55, 0.45])
        with rowB_r:
            st.markdown("<div class='compact-caption'>🔎 Search</div>", unsafe_allow_html=True)
            q = st.text_input("Search cases", placeholder="ID, customer, issue, owner, email, status, BU, region…", key="search_cases", label_visibility="collapsed")

        # Build final view
        view = filter_df_by_query(base, q)
        view["status"] = view["status"].fillna("Open").astype(str).str.strip().str.title()
        with rowB_l:
            st.markdown(f"<span class='counts-pill'>{_counts_bar(view)}</span>", unsafe_allow_html=True)

        # Kanban columns
        c1, c2, c3 = st.columns(3)
        counts = view['status'].value_counts()
        cols = {"Open": c1, "In Progress": c2, "Resolved": c3}
        model_for_view = train_model()

        for name, col in cols.items():
            with col:
                n = int(counts.get(name,0)); hdr = STATUS_COLORS[name]
                col.markdown(f"<div class='kanban-title' style='background:{hdr};'><span>{name}</span><span>({n})</span></div>", unsafe_allow_html=True)
                bucket = view[view["status"]==name]
                for _, row in bucket.iterrows():
                    try:
                        s  = (row.get("sentiment") or "neutral").lower()
                        u  = (row.get("urgency") or "normal").lower()
                        sv = (row.get("severity") or "minor").lower()
                        cr = (row.get("criticality") or "medium").lower()
                        likely = predict_escalation(model_for_view, s,u,sv,cr)

                        sev_color = SEVERITY_COLORS.get(sv, "#6b7280")
                        urg_color = URGENCY_COLORS.get(u, "#6b7280")
                        sent_color= {"negative":"#ef4444","positive":"#22c55e","neutral":"#f59e0b"}.get(s, "#6b7280")
                        esc_color = "#dc2626" if likely=="Yes" else "#6b7280"

                        case_id  = row.get('id','N/A')
                        customer = row.get('customer','Unknown')
                        summary  = summarize_issue_text(row.get('issue',''))
                        flag = "🚩" if likely=="Yes" else ""

                        # Age chip
                        try:
                            ts = pd.to_datetime(row.get("timestamp"))
                            dlt = datetime.datetime.now() - ts
                            days = dlt.days; hours, rem = divmod(dlt.seconds, 3600); minutes, _ = divmod(rem, 60)
                            age_str = f"{days}d {hours}h {minutes}m"
                            age_col = "#22c55e" if dlt.total_seconds()/3600 < 12 else "#f59e0b" if dlt.total_seconds()/3600 < 24 else "#ef4444"
                        except Exception:
                            age_str, age_col = "N/A", "#6b7280"

                        with st.expander(f"🆔 {case_id} — {customer} {flag}", expanded=False):
                            # Summary + Age
                            r0a, r0b = st.columns([0.75, 0.25])
                            with r0a:
                                st.markdown(f"<div class='summary'>{summary}</div>", unsafe_allow_html=True)
                            with r0b:
                                st.markdown(f"<div style='text-align:right;'><span class='age' style='background:{age_col};'>Age: {age_str}</span></div>", unsafe_allow_html=True)

                            # KPI panel
                            st.markdown("<div class='kpi-panel'>", unsafe_allow_html=True)
                            ka1, ka2, ka3 = st.columns(3)
                            with ka1:
                                st.markdown(f"<div>📛 <b>Severity</b> <span class='tag-pill' style='--c:{sev_color}; border-color:{sev_color}; color:{sev_color};'>{sv.capitalize()}</span></div>", unsafe_allow_html=True)
                            with ka2:
                                st.markdown(f"<div>⚡ <b>Urgency</b> <span class='tag-pill' style='--c:{urg_color}; border-color:{urg_color}; color:{urg_color};'>{'High' if u=='high' else 'Normal'}</span></div>", unsafe_allow_html=True)
                            with ka3:
                                st.markdown(f"<div>🎯 <b>Criticality</b> <span class='tag-pill' style='--c:#8b5cf6; border-color:#8b5cf6; color:#8b5cf6;'>{cr.capitalize()}</span></div>", unsafe_allow_html=True)

                            st.markdown("<div class='kpi-gap'></div>", unsafe_allow_html=True)

                            kb1, kb2, kb3 = st.columns(3)
                            with kb1:
                                st.markdown(f"<div>📂 <b>Category</b> <span class='tag-pill'>{(row.get('category') or 'other').capitalize()}</span></div>", unsafe_allow_html=True)
                            with kb2:
                                st.markdown(f"<div>💬 <b>Sentiment</b> <span class='tag-pill' style='--c:{sent_color}; border-color:{sent_color}; color:{sent_color};'>{s.capitalize()}</span></div>", unsafe_allow_html=True)
                            with kb3:
                                st.markdown(f"<div>📈 <b>Likely</b> <span class='tag-pill' style='--c:{esc_color}; border-color:{esc_color}; color:{esc_color};'>{likely}</span></div>", unsafe_allow_html=True)

                            st.markdown("</div>", unsafe_allow_html=True)

                            # Controls
                            st.markdown("<div class='controls-panel'>", unsafe_allow_html=True)
                            prefix = f"case_{case_id}"

                            # Row A: Status + Action Taken
                            ra1, ra2 = st.columns([1.0, 2.2])
                            with ra1:
                                current_status = (row.get("status") or "Open").strip().title()
                                new_status = st.selectbox(
                                    "Status",
                                    ["Open","In Progress","Resolved"],
                                    index=["Open","In Progress","Resolved"].index(current_status) if current_status in ["Open","In Progress","Resolved"] else 0,
                                    key=f"{prefix}_status",
                                )
                            with ra2:
                                action_taken = st.text_input("Action Taken", row.get("action_taken",""), key=f"{prefix}_action")

                            # Row B: Owner | Owner Email
                            rb1, rb2 = st.columns(2)
                            with rb1:
                                owner = st.text_input("Owner", row.get("owner",""), key=f"{prefix}_owner")
                            with rb2:
                                owner_email = st.text_input("Owner Email", row.get("owner_email",""), key=f"{prefix}_email")

                            # Row C: Save | N+1 Email | Escalate
                            rc1, rc2, rc3 = st.columns([0.9, 1.6, 1.0])
                            with rc1:
                                if st.button("💾 Save", key=f"{prefix}_save"):
                                    update_escalation_status(case_id, new_status, action_taken, owner, owner_email)
                                    st.success("✅ Saved")
                            with rc2:
                                n1_email = st.text_input("N+1 Email ID", key=f"{prefix}_n1")
                            with rc3:
                                if st.button("🚀 Escalate to N+1", key=f"{prefix}_n1btn"):
                                    update_escalation_status(
                                        case_id, new_status,
                                        action_taken or row.get("action_taken",""),
                                        owner or row.get("owner",""),
                                        n1_email
                                    )
                                    if n1_email:
                                        send_alert(f"Case {case_id} escalated to N+1.", via="email", recipient=n1_email)
                                    send_alert(f"Case {case_id} escalated to N+1.", via="teams")
                                    st.success("🚀 Escalated to N+1")

                            st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error rendering case #{row.get('id','Unknown')}: {e}")

    # ---------------- Tab 1: Likely ----------------
    with tabs[1]:
        st.subheader("🚩 Likely to Escalate")
        d = df_all.copy()
        if not d.empty:
            m = train_model()
            d["likely_calc"] = d.apply(lambda r: predict_escalation(
                m,
                (r.get("sentiment") or "neutral").lower(),
                (r.get("urgency") or "normal").lower(),
                (r.get("severity") or "minor").lower(),
                (r.get("criticality") or "medium").lower()
            ), axis=1)
            d = d[d["likely_calc"]=="Yes"]
        q2 = st.text_input("🔎 Search likely to escalate", placeholder="Search…")
        st.dataframe(filter_df_by_query(d, q2).sort_values(by="timestamp", ascending=False), use_container_width=True)

    # ---------------- Tab 2: Feedback ----------------
    with tabs[2]:
        st.subheader("🔁 Feedback & Retraining")
        d = fetch_escalations()
        if d.empty:
            st.info("No cases available.")
        else:
            d = d.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
            # show 3 cases per page
            per_page = 3
            if "fb_page" not in st.session_state: st.session_state.fb_page = 0
            total_pages = max(1, (len(d) + per_page - 1) // per_page)

            nav_c1, nav_c2, nav_c3 = st.columns([0.15, 0.7, 0.15])
            with nav_c1:
                if st.button("◀️ Prev", disabled=(st.session_state.fb_page==0)):
                    st.session_state.fb_page -= 1
            with nav_c2:
                st.markdown(f"<div style='text-align:center;color:#64748b;'>Page {st.session_state.fb_page+1} of {total_pages}</div>", unsafe_allow_html=True)
            with nav_c3:
                if st.button("Next ▶️", disabled=(st.session_state.fb_page >= total_pages-1)):
                    st.session_state.fb_page += 1

            start = st.session_state.fb_page * per_page
            slice_df = d.iloc[start:start+per_page]

            cc1, cc2, cc3 = st.columns(3)
            cols = [cc1, cc2, cc3]
            for (idx, r), col in zip(slice_df.iterrows(), cols):
                with col:
                    with st.expander(f"🆔 {r['id']}", expanded=True):
                        fb   = st.selectbox("Escalation Accuracy", ["Correct","Incorrect"], key=f"fb_{r['id']}")
                        sent = st.selectbox("Sentiment", ["positive","neutral","negative"], key=f"sent_{r['id']}")
                        crit = st.selectbox("Criticality", ["low","medium","high","urgent"], key=f"crit_{r['id']}")
                        notes= st.text_area("Notes", key=f"note_{r['id']}")
                        if st.button("Submit", key=f"btn_{r['id']}"):
                            owner_email_ = r.get("owner_email", EMAIL_USER)
                            update_escalation_status(r['id'], r.get("status","Open"),
                                                     r.get("action_taken",""), r.get("owner",""),
                                                     owner_email_, notes=notes, sentiment=sent, criticality=crit)
                            if owner_email_:
                                send_alert("Feedback recorded on your case.", via="email", recipient=owner_email_)
                            st.success("Feedback saved.")

            st.markdown("---")
            if st.button("🔁 Retrain Model"):
                st.info("Retraining model…")
                m = train_model()
                st.success("Model retrained.") if m else st.warning("Not enough data to retrain.")
                if m:
                    try: show_feature_importance(m)
                    except Exception: pass

    # ---------------- Tab 3: Summary Analytics ----------------
    with tabs[3]:
        st.subheader("📊 Summary Analytics")
        try: render_analytics()
        except Exception as e: st.info("Analytics module not fully configured."); st.exception(e)

    # ---------------- Tab 4: Help ----------------
    with tabs[4]:
        st.subheader("ℹ️ How this Dashboard Works")
        st.markdown("""
- **Header row:** Escalation View + SLA breach capsule + AI summary
- **Totals/Search row:** **Total (filtered)** pill on the left; **compact search** on the right
- **Kanban Columns:** Open (🟧), In Progress (🔵), Resolved (🟩)
- **Expander layout:**
  - Issue summary + age chip
  - KPI panel (2 rows): Severity, Urgency, Criticality, Category, Sentiment, Likely
  - Controls: **Status + Action Taken**, **Owner + Owner Email**, **Save + N+1 Email + Escalate to N+1**
- **Filters:** in the sidebar (Status, Severity, Sentiment, Category, BU, Region)
- **Notifications:** Compose & send to **Teams/Email** from sidebar; **WhatsApp/SMS** only for **Resolved**.
        """)

elif page == "🔥 SLA Heatmap":
    st.subheader("🔥 SLA Heatmap")
    try: render_sla_heatmap()
    except Exception as e: st.error(f"❌ SLA Heatmap failed: {type(e).__name__}: {str(e)}")

elif page == "🧠 Enhancements":
    try:
        from enhancement_dashboard import show_enhancement_dashboard
        show_enhancement_dashboard()
    except Exception as e:
        st.info("Enhancement dashboard not available."); st.exception(e)

elif page == "📈 Advanced Analytics":
    try: show_analytics_view()
    except Exception as e: st.error("❌ Failed to load analytics view."); st.exception(e)

elif page == "📊 BU/Region Trends":
    st.subheader("📊 Trends by BU & Region")
    d = fetch_escalations()
    if d.empty:
        st.info("No data yet.")
    else:
        # BU distribution with labels
        st.markdown("#### BU Distribution")
        bu_counts = d.get("bu_code", pd.Series(dtype=str)).value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,3))
        bu_counts.plot(kind="bar", ax=ax)
        for i, v in enumerate(bu_counts.values):
            ax.text(i, v + (max(bu_counts.values)*0.02 if len(bu_counts) else 0.5), str(v), ha='center', va='bottom', fontsize=10)
        ax.set_ylabel("Count"); ax.set_xlabel("BU Code"); ax.grid(axis='y', alpha=0.2)
        st.pyplot(fig, clear_figure=True)

        # Region distribution with labels
        st.markdown("#### Region Distribution")
        region_counts = d.get("region", pd.Series(dtype=str)).value_counts().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6,3))
        region_counts.plot(kind="bar", ax=ax2, color="#4ade80")
        for i, v in enumerate(region_counts.values):
            ax2.text(i, v + (max(region_counts.values)*0.02 if len(region_counts) else 0.5), str(v), ha='center', va='bottom', fontsize=10)
        ax2.set_ylabel("Count"); ax2.set_xlabel("Region"); ax2.grid(axis='y', alpha=0.2)
        st.pyplot(fig2, clear_figure=True)

elif page == "⚙️ Admin Tools":
    def show_admin_panel():
        st.title("⚙️ Admin Tools")
        if st.button("🔍 Validate DB Schema"):
            try: validate_escalation_schema(); st.success("✅ Schema validated and healed.")
            except Exception as e: st.error(f"❌ Schema validation failed: {e}")
        st.subheader("📄 Audit Log Preview")
        try:
            log_escalation_action("init","N/A","system","Initializing audit log table")
            conn = sqlite3.connect("escalations.db"); df = pd.read_sql("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100", conn); conn.close()
            st.dataframe(df)
        except Exception as e:
            st.warning("⚠️ Audit log not available."); st.exception(e)
        st.subheader("📝 Manual Audit Entry")
        with st.form("manual_log"):
            action = st.text_input("Action Type"); case_id = st.text_input("Case ID")
            user = st.text_input("User"); details = st.text_area("Details")
            if st.form_submit_button("Log Action"):
                try: log_escalation_action(action, case_id, user, details); st.success("✅ Action logged.")
                except Exception as e: st.error(f"❌ Failed to log action: {e}")
    try: show_admin_panel()
    except Exception as e: st.info("Admin tools not available."); st.exception(e)

# Background workers
if 'email_thread' not in st.session_state:
    t = threading.Thread(target=email_polling_job, daemon=True); t.start(); st.session_state['email_thread']=t

# Daily email scheduler
def send_daily_escalation_email():
    d = fetch_escalations(); e = d[d.get("likely_to_escalate", pd.Series(dtype=str)).astype(str).str.lower()=="yes"] if not d.empty else d
    if e.empty: return
    path = "daily_escalated_cases.xlsx"; e.to_excel(path, index=False)
    summary = f"""🔔 Daily Escalation Summary – {datetime.datetime.now():%Y-%m-%d}
Total Likely to Escalate Cases: {len(e)}
Open: {e[e['status'].astype(str).str.strip().str.title()=='Open'].shape[0]}
In Progress: {e[e['status'].astype(str).str.strip().str.title()=='In Progress'].shape[0]}
Resolved: {e[e['status'].astype(str).str.strip().str.title()=='Resolved'].shape[0]}
Please find the attached Excel file for full details."""
    try:
        msg = MIMEMultipart(); msg['Subject']="📊 Daily Escalated Cases Report"; msg['From']=EMAIL_USER or "no-reply@escalateai"; msg['To']=ALERT_RECIPIENT or (EMAIL_USER or "")
        msg.attach(MIMEText(summary,'plain'))
        with open(path,"rb") as f:
            part = MIMEBase('application','octet-stream'); part.set_payload(f.read()); encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{path}"'); msg.attach(part)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            if EMAIL_USER and EMAIL_PASS: s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
    except Exception as ex: print(f"❌ Failed to send daily email: {ex}")

def schedule_daily_email():
    schedule.every().day.at("09:00").do(send_daily_escalation_email)
    def run():
        while True:
            schedule.run_pending(); time.sleep(60)
    threading.Thread(target=run, daemon=True).start()

if 'daily_email_thread' not in st.session_state:
    schedule_daily_email(); st.session_state['daily_email_thread']=True
