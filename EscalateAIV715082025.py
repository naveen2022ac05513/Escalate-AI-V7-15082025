# EscalateAI_main.py
# --------------------------------------------------------------------
# EscalateAI — Customer Escalation Prediction & Management Tool
# --------------------------------------------------------------------
# This build focuses on your latest UI/analytics changes:
# • AI summary = Total (filtered) + Likely-to-Escalate (model) in the same row
#   with Escalation View and SLA pill
# • Search bar kept in previous position (compact, no label)
# • Removed the "Total" capsule under AI summary
# • Counts strip (Total | Open | In Progress | Resolved) retained and filtered
# • Advanced Analytics = 2×2 grid (Volume, Severity, Sentiment, Age Buckets)
# • BU/Region charts moved to the "BU & Region Trends" page only
# • WhatsApp/SMS restricted to Resolved, Teams/Email composers in sidebar
# • BU/Region filters in sidebar; robust duplicate detection; dev reset clears cache
# --------------------------------------------------------------------

import os, re, time, datetime, threading, hashlib, sqlite3, smtplib, requests, imaplib, email, traceback
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Optional: deeper duplicate detection
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
        classify_bu, bucketize_region, enrich_with_bu_region
    )
    _HAS_BUCKETIZER = True
except Exception:
    _HAS_BUCKETIZER = False

# Optional Enhancements dashboard (safe to miss)
try:
    from enhancement_dashboard import show_enhancement_dashboard
    _HAS_ENH = True
except Exception:
    _HAS_ENH = False

from dotenv import load_dotenv

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

# Twilio SMS (optional)
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

# UI Colors
STATUS_COLORS   = {"Open":"#f59e0b","In Progress":"#3b82f6","Resolved":"#22c55e"}  # Orange/Blue/Green
SEVERITY_COLORS = {"critical":"#ef4444","major":"#f59e0b","minor":"#10b981"}
URGENCY_COLORS  = {"high":"#dc2626","normal":"#16a34a"}

processed_email_uids_lock = threading.Lock()
global_seen_hashes = set()  # keep as a set; we only mutate it

# ---------------- DB helpers ----------------
def ensure_schema():
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS escalations (
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
            duplicate_of TEXT,
            bu_code TEXT,
            bu_name TEXT,
            region TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS processed_hashes (
            hash TEXT PRIMARY KEY, first_seen TEXT
        )''')
        conn.commit()
    except Exception:
        traceback.print_exc()
    finally:
        try: conn.close()
        except Exception: pass

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

def fetch_escalations() -> pd.DataFrame:
    ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception as e:
        st.error(f"Error reading escalations table: {e}"); df = pd.DataFrame()
    finally:
        conn.close()
    return df

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
    except Exception:
        pass
    finally:
        try: conn.close()
        except Exception: pass

# --------------- Duplicate detection ---------------
def _cosine_sim(a: str, b: str) -> float:
    a, b = _normalize_text(a), _normalize_text(b)
    if not a or not b: return 0.0
    if _TFIDF_AVAILABLE:
        try:
            vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
            X = vec.fit_transform([a,b]); return float(_cosine(X[0],X[1]))
        except Exception:
            pass
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
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    if df.empty: return (False,None,0.0)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_window)
        df = df[df['timestamp'].isna() | (df['timestamp'] >= cutoff)]
    except Exception:
        pass
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

# --------------- Insert / Update ---------------
def insert_escalation(customer, issue, sentiment, urgency, severity,
                      criticality, category, escalation_flag, likely_to_escalate="No",
                      owner_email="", issue_hash=None, duplicate_of=None,
                      bu_code=None, bu_name=None, region=None):
    ensure_schema()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    priority = "high" if str(severity).lower()=="critical" or str(urgency).lower()=="high" else "normal"
    issue_hash = issue_hash or generate_issue_hash(issue)

    # BU & Region classification
    if _HAS_BUCKETIZER:
        try:
            bcode, bname = classify_bu(issue)
        except Exception:
            bcode, bname = (bu_code or "OTHER", bu_name or "Other / Unclassified")
    else:
        bcode, bname = (bu_code or "OTHER", bu_name or "Other / Unclassified")

    if region is None:
        region = "Others"

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
            likely_to_escalate, "", "", "", duplicate_of,
            bcode, bname, region
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
                           region=None):
    h = generate_issue_hash(issue)
    if _processed_hash_exists(h):
        return (False,None,1.0)
    is_dup, dup_id, score = find_duplicate(issue, customer=customer)
    if is_dup and dup_id:
        _mark_processed_hash(h)
        return (False,dup_id,score)
    new_id = insert_escalation(customer, issue, sentiment, urgency, severity, criticality,
                               category, escalation_flag, likely_to_escalate, owner_email,
                               issue_hash=h, duplicate_of=None, region=region)
    _mark_processed_hash(h)
    return (True,new_id,1.0)

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
            if not (recipient or ALERT_RECIPIENT):
                st.warning("No email recipient configured."); return
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['Subject']=EMAIL_SUBJECT
            msg['From']=EMAIL_USER or "no-reply@escalateai"
            msg['To']=recipient if recipient else ALERT_RECIPIENT
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
                s.starttls()
                if EMAIL_USER and EMAIL_PASS: s.login(EMAIL_USER, EMAIL_PASS)
                s.sendmail(msg['From'], [msg['To']], msg.as_string())
        except Exception as e: st.error(f"Email alert failed: {e}")
    elif via=="teams":
        try:
            if not TEAMS_WEBHOOK:
                st.error("MS Teams webhook URL is not configured."); return
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
            add_or_skip_escalation(e["customer"], e["issue"], s,u,sev,c,cat,esc, likely)
        time.sleep(60)

# ---------- Analytics helpers (Altair) ----------
def _bar_with_labels(
    df, x_field: str, y_field: str,
    title: str | None = None,
    height: int = 240,
    color_field: str | None = None,
    color_domain: list[str] | None = None,
    color_range: list[str] | None = None
):
    if df is None or df.empty:
        return None
    df = df.copy()
    if x_field in df.columns:
        df[x_field] = df[x_field].fillna("Unknown").astype(str)
    if y_field in df.columns:
        df[y_field] = pd.to_numeric(df[y_field], errors="coerce").fillna(0).astype(int)

    base = alt.Chart(df).encode(
        x=alt.X(f"{x_field}:N", sort="-y", title=None),
        y=alt.Y(f"{y_field}:Q", title=None)
    )
    color_enc = (
        alt.Color(f"{color_field}:N",
                  scale=alt.Scale(domain=color_domain, range=color_range),
                  legend=None)
        if color_field else alt.value("#4f46e5")
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(color=color_enc)
    labels = base.mark_text(dy=-6).encode(
        text=alt.Text(f"{y_field}:Q", format="d"),
        color=alt.value("#111827")
    )
    chart = alt.layer(bars, labels).properties(height=height)
    if title is not None:
        chart = chart.properties(title=str(title))
    return chart.configure_view(strokeWidth=0)

def _line_with_points(df, x_field: str, y_field: str, title: str):
    if df is None or df.empty:
        return None
    df = df.copy()
    line = alt.Chart(df).mark_line().encode(
        x=alt.X(f"{x_field}:T", title=None),
        y=alt.Y(f"{y_field}:Q", title=None)
    )
    pts = alt.Chart(df).mark_point().encode(
        x=alt.X(f"{x_field}:T", title=None),
        y=alt.Y(f"{y_field}:Q", title=None)
    )
    ch = (line + pts).properties(height=220)
    if title is not None:
        ch = ch.properties(title=str(title))
    return ch.configure_view(strokeWidth=0)

def show_analytics_view():
    """Advanced Analytics 2×2 grid: Volume, Severity, Sentiment, Age Buckets (no BU/Region here)."""
    df = fetch_escalations()
    st.title("📈 Advanced Analytics")
    if df.empty:
        st.warning("⚠️ No escalation data available."); return

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["severity"]  = df.get("severity",  "").astype(str).str.lower().replace({"nan":"unknown"})
    df["sentiment"] = df.get("sentiment", "").astype(str).str.lower().replace({"nan":"unknown"})

    # Volume
    vol = (df.assign(day=df["timestamp"].dt.date)
             .groupby("day", dropna=False).size()
             .reset_index(name="count"))
    # Severity
    sev = df.groupby("severity", dropna=False).size().reset_index(name="count")
    sev_order  = ["minor","major","critical","unknown"]
    sev_colors = ["#10b981","#f59e0b","#ef4444","#9ca3af"]
    # Sentiment
    sen = df.groupby("sentiment", dropna=False).size().reset_index(name="count")
    sent_order  = ["negative","neutral","positive","unknown"]
    sent_colors = ["#ef4444","#f59e0b","#22c55e","#9ca3af"]
    # Age Buckets
    df["age_days"] = (pd.Timestamp.now() - df["timestamp"]).dt.days
    df["age_bucket"] = pd.cut(df["age_days"],
                              bins=[-1,3,7,14,30,90,9999],
                              labels=["0–3d","4–7d","8–14d","15–30d","31–90d",">90d"])
    age = df.groupby("age_bucket", dropna=False).size().reset_index(name="count")

    # Layout 2×2
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        ch = _line_with_points(vol.rename(columns={"day":"date"}), "date", "count", "Escalation Volume (daily)")
        if ch is not None: st.altair_chart(ch, use_container_width=True)
    with r1c2:
        ch = _bar_with_labels(sev, "severity", "count", "Severity (count)", 240, "severity", sev_order, sev_colors)
        if ch is not None: st.altair_chart(ch, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        ch = _bar_with_labels(sen, "sentiment", "count", "Sentiment (count)", 240, "sentiment", sent_order, sent_colors)
        if ch is not None: st.altair_chart(ch, use_container_width=True)
    with r2c2:
        ch = _bar_with_labels(age, "age_bucket", "count", "Age Buckets (count)", 240)
        if ch is not None: st.altair_chart(ch, use_container_width=True)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Escalation Management", layout="wide")
ensure_schema()

# Styles
st.markdown("""
<style>
  .sticky-header{position:sticky;top:0;z-index:999;background:linear-gradient(135deg,#0ea5e9 0%,#7c3aed 100%);
    padding:12px 16px;border-radius:0 0 12px 12px;box-shadow:0 8px 20px rgba(0,0,0,.12);}
  .sticky-header h1{color:#fff;margin:0;text-align:center;font-size:30px;line-height:1.2;} /* 1.5× */

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

  .kv{font-size:12px;margin:2px 0;white-space:nowrap;}
  .aisum{background:#0b1220;color:#e5f2ff;padding:10px 12px;border-radius:10px;
         box-shadow:0 6px 14px rgba(0,0,0,.10);font-size:13px;}
  .sla-pill{display:inline-block;padding:4px 8px;border-radius:999px;background:#ef4444;color:#fff;font-weight:600;font-size:12px;}

  .kpi-panel{margin-top:0 !important;background:transparent !important;border:0 !important;box-shadow:none !important;padding:0 !important;}
  .kpi-gap{height:22px !important;}

  .controls-panel{background:#fff;border:0;border-radius:12px;padding:10px 0 2px 0;margin:6px 0 2px 0;}
  div[data-testid="stTextInput"]  label,
  div[data-testid="stTextArea"]   label,
  div[data-testid="stSelectbox"]  label {font-size:13px !important;font-weight:600 !important;color:#475569 !important;margin-bottom:4px !important;}

  div[data-testid="stTextInput"] input,
  div[data-testid="stTextArea"] textarea {background:#f3f4f6 !important;border:1px solid #e5e7eb !important;border-radius:8px !important;height:40px !important;padding:8px 10px !important;}
  div[data-testid="stSelectbox"] div[role="combobox"]{background:#f3f4f6 !important;border:1px solid #e5e7eb !important;border-radius:8px !important;min-height:40px !important;padding:6px 10px !important;align-items:center !important;}

  .controls-panel .stButton>button{height:40px !important;border-radius:10px !important;padding:0 14px !important;}

  .tag-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600;
            border:1px solid var(--c,#cbd5e1);color:var(--c,#334155);background:#fff;white-space:nowrap;}
</style>
<div class="sticky-header"><h1>🚨 EscalateAI – AI Based Customer Escalation Prediction & Management Tool</h1></div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Main Dashboard", "📈 Advanced Analytics", "📈 BU & Region Trends",
    "🔥 SLA Heatmap", "🧠 Enhancements", "⚙️ Admin Tools", "ℹ️ User Guide"
])

# Sidebar: Email Integration
st.sidebar.markdown("### 📩 Email Integration")
if st.sidebar.button("Fetch Emails"):
    mails = parse_emails(); m = train_model()
    added, skipped = 0, 0
    for e in mails:
        s,u,sev,c,cat,esc = analyze_issue(e["issue"])
        likely = predict_escalation(m,s,u,sev,c)
        created, _, _ = add_or_skip_escalation(e["customer"], e["issue"], s,u,sev,c,cat,esc, likely)
        if created: added += 1
        else: skipped += 1
    st.sidebar.success(f"✅ Processed {len(mails)} unread. Inserted {added}, skipped {skipped} duplicate(s).")

# Sidebar: Upload
st.sidebar.header("📁 Upload Escalation Sheet")
uploaded = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded:
    try:
        df_x = pd.read_excel(uploaded); st.sidebar.success("✅ Excel loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Read failed: {e}"); st.stop()
    missing = [c for c in ["Customer","Issue"] if c not in df_x.columns]
    if missing:
        st.sidebar.error("Missing required columns: " + ", ".join(missing)); st.stop()
    if st.sidebar.button("🔍 Analyze & Insert"):
        m = train_model(); ok, dups = 0, 0
        for _, r in df_x.iterrows():
            issue = str(r.get("Issue","")).strip(); cust = str(r.get("Customer","Unknown")).strip()
            if not issue: continue
            text = summarize_issue_text(issue)
            s,u,sev,c,cat,esc = analyze_issue(text); likely = predict_escalation(m,s,u,sev,c)
            region_guess = "Others"
            created, _, _ = add_or_skip_escalation(cust, text, s,u,sev,c,cat,esc, likely, region=region_guess)
            ok += int(created); dups += int(not created)
        st.sidebar.success(f"🎯 Inserted {ok}, skipped {dups} duplicate(s).")

# Sidebar: Filters (affect all views)
st.sidebar.markdown("### 🔍 Filters")
status_opt    = st.sidebar.selectbox("Status",   ["All","Open","In Progress","Resolved"], index=0)
severity_opt  = st.sidebar.selectbox("Severity", ["All","minor","major","critical"], index=0)
sentiment_opt = st.sidebar.selectbox("Sentiment",["All","positive","neutral","negative"], index=0)
category_opt  = st.sidebar.selectbox("Category", ["All","technical","support","dissatisfaction","safety","business","other"], index=0)
bu_opt        = st.sidebar.selectbox("BU", ["All","SPIBS","PPIBS","PSIBS","IDIBS","BMS","H&D","A2E","Solar","OTHER"], index=0)
region_opt    = st.sidebar.selectbox("Region", ["All","North","East","South","West","NC","Others"], index=0)

# Sidebar: Notifications — MS Teams & Email
st.sidebar.markdown("### 📣 MS Teams & Email")
notify_msg = st.sidebar.text_area("Message", "Test notification from EscalateAI")
email_to   = st.sidebar.text_input("Email recipient", ALERT_RECIPIENT or "")
col_te, col_em = st.sidebar.columns(2)
with col_te:
    if st.button("Send to Teams"):
        send_alert(notify_msg, via="teams")
        st.sidebar.success("✅ Sent to Teams")
with col_em:
    if st.button("Send Email"):
        send_alert(notify_msg, via="email", recipient=email_to if email_to else None)
        st.sidebar.success("✅ Email sent")

# Sidebar: WhatsApp & SMS — only for Resolved
st.sidebar.markdown("### 📲 WhatsApp & SMS (Resolved only)")
df_notify = fetch_escalations()
if not df_notify.empty:
    st_notify = df_notify.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.title()
    df_resolved_only = df_notify[st_notify == "Resolved"]
else:
    df_resolved_only = pd.DataFrame()

if not df_resolved_only.empty:
    esc_id = st.sidebar.selectbox("🔢 Resolved Escalation ID", df_resolved_only["id"].astype(str).tolist())
    phone  = st.sidebar.text_input("📞 Phone Number", "+91", help="Include country code (e.g., +91)")
    msg2   = st.sidebar.text_area("📨 Message (Resolved)", f"Your case {esc_id} is resolved. Thank you!")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Send WhatsApp"):
            try:
                ok = False  # plug your WhatsApp sender here
                st.sidebar.success(f"✅ WhatsApp sent to {phone}") if ok else st.sidebar.error("❌ WhatsApp API not configured")
            except Exception as e:
                st.sidebar.error(f"❌ WhatsApp send failed: {e}")
    with c2:
        if st.button("Send SMS"):
            st.sidebar.success(f"✅ SMS sent to {phone}") if send_sms(phone, msg2) else None
else:
    st.sidebar.info("No Resolved cases for WhatsApp/SMS.")

# Sidebar: Developer
st.sidebar.markdown("### 🛠️ Developer")
def send_daily_escalation_email():
    d = fetch_escalations()
    if d.empty:
        st.sidebar.info("No data available."); return
    e = d[d.get("likely_to_escalate", pd.Series(dtype=str)).astype(str).str.lower()=="yes"]
    if e.empty:
        st.sidebar.info("No 'Likely to Escalate' cases for daily email."); return
    path = "daily_escalated_cases.xlsx"; e.to_excel(path, index=False)
    summary = f"""🔔 Daily Escalation Summary – {datetime.datetime.now():%Y-%m-%d}
Total Likely to Escalate Cases: {len(e)}
Open: {e[e['status'].astype(str).str.strip().str.title()=='Open'].shape[0]}
In Progress: {e[e['status'].astype(str).str.strip().str.title()=='In Progress'].shape[0]}
Resolved: {e[e['status'].astype(str).str.strip().str.title()=='Resolved'].shape[0]}
See attached for details."""
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
        st.sidebar.success("✅ Daily escalation email sent.")
    except Exception as ex:
        st.sidebar.error(f"❌ Failed to send daily email: {ex}")

if st.sidebar.button("Send Daily Email"):
    send_daily_escalation_email()

if st.sidebar.checkbox("🧪 View Raw Database"):
    st.sidebar.dataframe(fetch_escalations(), use_container_width=True)

def reset_database():
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS escalations")
        cur.execute("DROP TABLE IF EXISTS processed_hashes")
        conn.commit(); conn.close()
    except Exception as e:
        st.sidebar.error(f"Reset failed: {e}"); return
    ensure_schema()
    global_seen_hashes.clear()  # clear in-memory fingerprint cache too
    st.sidebar.warning("Database reset. (Tables recreated; duplicate memory cleared.)")

if st.sidebar.button("🗑️ Reset Database (Dev Only)"):
    reset_database()

auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 10, 60, 30)
if auto_refresh:
    time.sleep(refresh_interval); st.rerun()
if st.sidebar.button("🔁 Manual Refresh"):
    st.rerun()

# Helpers
def filter_df_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query or df.empty: return df
    q = str(query).strip().lower()
    cols = ['id','customer','issue','owner','action_owner','owner_email','category','severity','sentiment','status','bu_code','region']
    present = [c for c in cols if c in df.columns]
    combined = df[present].astype(str).apply(lambda s: s.str.lower()).agg(' '.join, axis=1)
    return df[combined.str.contains(q, na=False, regex=False)]

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["status"] = out.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.title()
    if status_opt != "All":    out = out[out["status"] == status_opt]
    if severity_opt != "All":  out = out[out.get("severity", "").astype(str).str.lower() == severity_opt.lower()]
    if sentiment_opt != "All": out = out[out.get("sentiment","").astype(str).str.lower() == sentiment_opt.lower()]
    if category_opt != "All":  out = out[out.get("category","").astype(str).str.lower() == category_opt.lower()]
    if bu_opt != "All":        out = out[out.get("bu_code","").astype(str) == bu_opt]
    if region_opt != "All":    out = out[out.get("region","").astype(str) == region_opt]
    return out

# ---------------- Routing ----------------
if page == "📊 Main Dashboard":
    df_all = fetch_escalations()
    df_all['timestamp'] = pd.to_datetime(df_all.get('timestamp'), errors='coerce')

    tabs = st.tabs(["🗃️ All Cases","🚩 Likely to Escalate","🔁 Feedback & Retraining","📊 Summary Analytics","ℹ️ How this Dashboard Works"])

    # ---------------- Tab 0: All Cases ----------------
    with tabs[0]:
        st.subheader("📊 Escalation Kanban Board — All Cases")

        # Apply sidebar filters first
        filt = apply_filters(df_all)

        # Precompute model-based "likely" on the filtered pool for AI Summary
        model_for_view = train_model()
        if not filt.empty:
            def _pred_row_summary(r):
                return predict_escalation(
                    model_for_view,
                    (str(r.get("sentiment") or "neutral")).lower(),
                    (str(r.get("urgency") or "normal")).lower(),
                    (str(r.get("severity") or "minor")).lower(),
                    (str(r.get("criticality") or "medium")).lower(),
                )
            likely_count = filt.apply(_pred_row_summary, axis=1).eq("Yes").sum()
        else:
            likely_count = 0

        # Top ribbon: Escalation View + SLA pill + AI Summary (same row)
        top_a1, top_a2, top_a3 = st.columns([0.50, 0.15, 0.35])
        with top_a1:
            view_radio = st.radio("Escalation View", ["All", "Likely to Escalate", "Not Likely", "SLA Breach"], horizontal=True)
        with top_a2:
            _x = filt.copy()
            _x['timestamp'] = pd.to_datetime(_x['timestamp'], errors='coerce')
            sla_breaches = _x[
                (_x.get('status', pd.Series(dtype=str)).astype(str).str.strip().str.title()!='Resolved')
                & (_x.get('priority', pd.Series(dtype=str)).astype(str).str.lower()=='high')
                & ((datetime.datetime.now()-_x['timestamp']) > datetime.timedelta(minutes=10))
            ]
            st.markdown(f"<span class='sla-pill'>⏱️ {len(sla_breaches)} SLA breach(s)</span>", unsafe_allow_html=True)
        with top_a3:
            ai_text = f"Total (filtered): <b>{len(filt)}</b> &nbsp;|&nbsp; Likely to Escalate: <b>{likely_count}</b>"
            st.markdown(f"<div class='aisum'><b>🧠 AI Summary</b><br>{ai_text}</div>", unsafe_allow_html=True)

        # Second ribbon: Search bar ONLY (same position as before; compact)
        top_b1, _ = st.columns([0.70, 0.30])
        with top_b1:
            q = st.text_input("", key="search_cases", placeholder="Search cases (ID, customer, issue, owner, email, status, BU, region…)", label_visibility="collapsed")

        # Apply search
        view = filter_df_by_query(filt.copy(), q)
        view["status"] = view.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.title()

        # KPI counts strip (reflecting filters+search) — SAME as before
        counts = view['status'].value_counts() if not view.empty else pd.Series(dtype=int)
        total = len(view)
        open_c = int(counts.get("Open",0)); prog_c = int(counts.get("In Progress",0)); res_c = int(counts.get("Resolved",0))
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total", total)
        k2.metric("Open", open_c)
        k3.metric("In Progress", prog_c)
        k4.metric("Resolved", res_c)

        # Escalation view refinement
        if view_radio == "SLA Breach":
            _y = view.copy()
            _y['timestamp'] = pd.to_datetime(_y['timestamp'], errors='coerce')
            view = _y[
                (_y['status']!='Resolved')
                & (_y.get('priority', pd.Series(dtype=str)).astype(str).str.lower()=='high')
                & ((datetime.datetime.now()-_y['timestamp']) > datetime.timedelta(minutes=10))
            ]
        elif view_radio in ("Likely to Escalate", "Not Likely") and not view.empty:
            def _pred_row(r):
                return predict_escalation(
                    model_for_view,
                    (str(r.get("sentiment") or "neutral")).lower(),
                    (str(r.get("urgency") or "normal")).lower(),
                    (str(r.get("severity") or "minor")).lower(),
                    (str(r.get("criticality") or "medium")).lower(),
                )
            view = view.copy()
            view["likely_calc"] = view.apply(_pred_row, axis=1)
            view = view[view["likely_calc"]=="Yes"] if view_radio=="Likely to Escalate" else view[view["likely_calc"]!="Yes"]

        # Kanban columns
        c1, c2, c3 = st.columns(3)
        cols = {"Open": c1, "In Progress": c2, "Resolved": c3}
        counts = view['status'].value_counts() if not view.empty else pd.Series(dtype=int)
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

                            # KPI PANEL (2 rows)
                            st.markdown("<div class='kpi-panel'>", unsafe_allow_html=True)
                            ka1, ka2, ka3 = st.columns(3)
                            with ka1:
                                st.markdown(f"<div class='kv'>📛 <b>Severity</b> <span class='tag-pill' style='--c:{sev_color}; border-color:{sev_color}; color:{sev_color};'>{sv.capitalize()}</span></div>", unsafe_allow_html=True)
                            with ka2:
                                st.markdown(f"<div class='kv'>⚡ <b>Urgency</b> <span class='tag-pill' style='--c:{urg_color}; border-color:{urg_color}; color:{urg_color};'>{'High' if u=='high' else 'Normal'}</span></div>", unsafe_allow_html=True)
                            with ka3:
                                st.markdown(f"<div class='kv'>🎯 <b>Criticality</b> <span class='tag-pill' style='--c:#8b5cf6; border-color:#8b5cf6; color:#8b5cf6;'>{cr.capitalize()}</span></div>", unsafe_allow_html=True)

                            st.markdown("<div class='kpi-gap'></div>", unsafe_allow_html=True)
                            kb1, kb2, kb3 = st.columns(3)
                            with kb1:
                                st.markdown(f"<div class='kv'>📂 <b>Category</b> <span class='tag-pill'>{(row.get('category') or 'other').capitalize()}</span></div>", unsafe_allow_html=True)
                            with kb2:
                                st.markdown(f"<div class='kv'>💬 <b>Sentiment</b> <span class='tag-pill' style='--c:{sent_color}; border-color:{sent_color}; color:{sent_color};'>{s.capitalize()}</span></div>", unsafe_allow_html=True)
                            with kb3:
                                st.markdown(f"<div class='kv'>📈 <b>Likely</b> <span class='tag-pill' style='--c:{esc_color}; border-color:{esc_color}; color:{esc_color};'>{likely}</span></div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # CONTROLS
                            st.markdown("<div class='controls-panel'>", unsafe_allow_html=True)
                            prefix = f"case_{case_id}"

                            # Row A: Status | Action Taken
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
                            rc1, rc2, rc3 = st.columns([0.9, 1.6, 1.2])
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

    # ---------------- Tab 1: Likely to Escalate ----------------
    with tabs[1]:
        st.subheader("🚩 Likely to Escalate")
        d = apply_filters(df_all)
        if not d.empty:
            m = train_model()
            d["likely_calc"] = d.apply(lambda r: predict_escalation(
                m,
                (str(r.get("sentiment") or "neutral")).lower(),
                (str(r.get("urgency") or "normal")).lower(),
                (str(r.get("severity") or "minor")).lower(),
                (str(r.get("criticality") or "medium")).lower()
            ), axis=1)
            d = d[d["likely_calc"]=="Yes"]
        q2 = st.text_input("", key="search_likely", placeholder="Search…", label_visibility="collapsed")
        st.dataframe(filter_df_by_query(d, q2).sort_values(by="timestamp", ascending=False), use_container_width=True)

    # ---------------- Tab 2: Feedback & Retraining ----------------
    with tabs[2]:
        st.subheader("🔁 Feedback & Retraining (18 per page)")
        d = apply_filters(fetch_escalations())
        if d.empty:
            st.info("No cases available.")
        else:
            d = d.sort_values(by="timestamp", ascending=False)
            per_page = 18
            total_pages = max(1, int(np.ceil(len(d)/per_page)))
            page_idx = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            start = (page_idx-1)*per_page; end = start+per_page
            page_df = d.iloc[start:end]

            cards = list(page_df.to_dict("records"))
            rows = [cards[i:i+3] for i in range(0, len(cards), 3)]
            for row_cards in rows:
                cols = st.columns(3)
                for c, rec in zip(cols, row_cards):
                    with c:
                        with st.expander(f"🆔 {rec['id']} — {rec.get('customer','Unknown')}", expanded=False):
                            fb   = st.selectbox("Escalation Accuracy", ["Correct","Incorrect"], key=f"fb_{rec['id']}")
                            sent = st.selectbox("Sentiment", ["positive","neutral","negative"], key=f"sent_{rec['id']}")
                            crit = st.selectbox("Criticality", ["low","medium","high","urgent"], key=f"crit_{rec['id']}")
                            notes= st.text_area("Notes", key=f"note_{rec['id']}")
                            if st.button("Submit", key=f"btn_{rec['id']}"):
                                owner_email_ = rec.get("owner_email", EMAIL_USER)
                                update_escalation_status(rec['id'], rec.get("status","Open"),
                                                         rec.get("action_taken",""), rec.get("owner",""),
                                                         owner_email_, notes=notes, sentiment=sent, criticality=crit)
                                if owner_email_: send_alert("Feedback recorded on your case.", via="email", recipient=owner_email_)
                                st.success("Feedback saved.")
            st.caption(f"Page {page_idx} / {total_pages}")

    # ---------------- Tab 3: Summary Analytics ----------------
    with tabs[3]:
        st.subheader("📊 Summary Analytics")
        d = apply_filters(df_all)
        counts = d.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.title().value_counts()
        status_df = pd.DataFrame({"Status": counts.index, "Count": counts.values})
        if not status_df.empty:
            color_map = {"Open":"#f59e0b","In Progress":"#3b82f6","Resolved":"#22c55e"}
            status_df["color"] = status_df["Status"].map(color_map).fillna("#64748b")
            ch = alt.Chart(status_df).mark_bar().encode(
                x=alt.X("Status:N", sort=None, title=None),
                y=alt.Y("Count:Q", title=None),
                color=alt.Color("Status:N", scale=None)
            ).properties(height=220, title="Cases by Status")
            labels = alt.Chart(status_df).mark_text(dy=-6, color="#111827").encode(x="Status:N", y="Count:Q", text="Count:Q")
            st.altair_chart((ch+labels), use_container_width=True)

        # Also show the same 2×2 grid here for quick glance (no BU/Region)
        try:
            show_analytics_view()
        except Exception as e:
            st.error(f"❌ Failed to load analytics view: {e}")

    # ---------------- Tab 4: How it Works / User Guide ----------------
    with tabs[4]:
        st.subheader("ℹ️ How this Dashboard Works")
        st.markdown("""
**At a glance**
- **Sidebar filters** (Status/Severity/Sentiment/Category/BU/Region) narrow *all* tabs.
- **Escalation View** toggles All / Likely / Not Likely / SLA Breach.
- **SLA pill** shows unresolved, high-priority cases older than 10 minutes.
- **Search** is compact and searches across ID, customer, issue, owner, email, status, BU, region.
- **Counts strip** (Total/Open/In Progress/Resolved) always reflects filters + search.
- **Kanban** headers are color-coded: orange (Open), blue (In Progress), green (Resolved).

**Inside each case**
1) **Summary** + **Age** chip  
2) **KPI rows:** Severity, Urgency, Criticality, Category, Sentiment, Likely  
3) **Controls:**  
   - Row A: **Status** + **Action Taken**  
   - Row B: **Owner** + **Owner Email**  
   - Row C: **Save** + **N+1 Email** + **Escalate to N+1**

**Notifications**
- Send **MS Teams** and **Email** from the sidebar.  
- **WhatsApp/SMS** available for **Resolved** only.

**Duplicates**
- Normalization + TF-IDF (fallback difflib) to prevent re-inserts.  
- **Reset Database** clears tables and the in-memory duplicate cache.

**Analytics**
- **Advanced Analytics** shows a 2×2 grid of Volume, Severity, Sentiment, and Age Buckets.  
- **BU & Region Trends** dedicates charts for BU and Region distributions.
        """)

elif page == "📈 Advanced Analytics":
    try:
        show_analytics_view()
    except Exception as e:
        st.error(f"❌ Failed to load analytics view: {e}")

elif page == "📈 BU & Region Trends":
    st.subheader("📈 BU & Region Trends")
    df = apply_filters(fetch_escalations())
    if df.empty:
        st.info("No data yet.")
    else:
        # BU distribution
        st.markdown("**🏷️ BU Distribution**")
        bu_order  = ["SPIBS","PPIBS","PSIBS","IDIBS","BMS","H&D","A2E","Solar","OTHER"]
        bu_colors = ["#2563eb","#16a34a","#f59e0b","#9333ea","#06b6d4","#f97316","#a855f7","#22c55e","#94a3b8"]
        bu = df.get("bu_code", pd.Series(dtype=str)).astype(str)
        bu_ct = bu.value_counts().reindex(bu_order, fill_value=0)
        bu_df = pd.DataFrame({"bu_code": bu_ct.index, "count": bu_ct.values})
        ch = _bar_with_labels(bu_df, "bu_code","count","BU (count)",240,"bu_code",bu_order,bu_colors)
        if ch is not None: st.altair_chart(ch, use_container_width=True)

        # Region distribution
        st.markdown("**🗺️ Region Distribution**")
        reg_order  = ["North","East","South","West","NC","Others"]
        reg_colors = ["#2563eb","#d97706","#059669","#9333ea","#dc2626","#64748b"]
        reg = df.get("region", pd.Series(dtype=str)).astype(str)
        reg_ct = reg.value_counts().reindex(reg_order, fill_value=0)
        reg_df = pd.DataFrame({"region": reg_ct.index, "count": reg_ct.values})
        ch = _bar_with_labels(reg_df, "region","count","Region (count)",240,"region",reg_order,reg_colors)
        if ch is not None: st.altair_chart(ch, use_container_width=True)

elif page == "🔥 SLA Heatmap":
    st.subheader("🔥 SLA Heatmap")
    st.info("Hook your custom SLA heatmap here (e.g., from enhancements).")

elif page == "🧠 Enhancements":
    if _HAS_ENH:
        try:
            show_enhancement_dashboard()
        except Exception as e:
            st.error(f"Enhancement dashboard error: {e}")
    else:
        st.info("Enhancement dashboard not available (module not found).")

elif page == "⚙️ Admin Tools":
    st.title("⚙️ Admin Tools")
    st.write("Use the Developer section in the sidebar for maintenance tasks.")

elif page == "ℹ️ User Guide":
    st.title("ℹ️ User Guide")
    st.markdown("""
**Purpose**  
EscalateAI predicts and manages customer escalations from email uploads or Excel imports, deduplicates similar issues, and routes updates/alerts via Teams, Email, WhatsApp, and SMS.

**Key Features**  
- **Kanban** with Open/Progress/Resolved and age chips  
- **Filters**: Status, Severity, Sentiment, Category, BU, Region  
- **AI Likelihood** and **SLA Breach** view  
- **Feedback & Retraining** grid (18 per page)  
- **Analytics** (2×2 grid) + **BU/Region Trends**  
- **Notifications**: Teams, Email (any); WhatsApp/SMS (Resolved only)  
- **Dev tools**: daily email, DB viewer, safe reset

**How to Use**  
1. Use **Sidebar → Upload** to load Excel (needs `Customer`, `Issue`).  
2. Click **Analyze & Insert** → dedupe + auto-tag.  
3. Filter in sidebar; search in the main.  
4. Update a case inside its expander; save; escalate if needed.  
5. Use **Advanced Analytics** and **BU & Region Trends** for insights.  
6. Set Teams webhook and email creds in `.env` for notifications.
    """)
