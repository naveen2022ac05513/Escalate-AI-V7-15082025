# enhancement_dashboard.py
# Minimal Enhancement Dashboard (no optional libs)

import os, sqlite3
import pandas as pd
import streamlit as st

DB_PATH = os.getenv("ESCALATEAI_DB_PATH") or "escalations.db"

def _load_escalations() -> pd.DataFrame:
    try:
        if not os.path.exists(DB_PATH):
            return pd.DataFrame()
        with sqlite3.connect(DB_PATH) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='escalations'"
            ).fetchone()
            if not exists:
                return pd.DataFrame()
            df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        return pd.DataFrame()

    # Normalize common fields
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["status","severity","urgency","criticality","sentiment","category","likely_to_escalate"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Title-case status for consistent counting
    if "status" in df.columns:
        df["status"] = df["status"].str.strip().str.title()

    return df

def _safe_count(df: pd.DataFrame, col: str, value: str) -> int:
    if col not in df.columns:
        return 0
    return int((df[col].astype(str).str.strip().str.title() == value).sum())

def show_enhancement_dashboard():
    st.header("🧠 Enhancements")

    df = _load_escalations()
    if df.empty:
        st.info("No data yet.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Cases", len(df))
    with c2: st.metric("Open", _safe_count(df, "status", "Open"))
    with c3: st.metric("In Progress", _safe_count(df, "status", "In Progress"))
    with c4: st.metric("Resolved", _safe_count(df, "status", "Resolved"))

    st.divider()
    st.subheader("📦 Distributions (tables)")

    left, right = st.columns(2)
    with left:
        if "severity" in df.columns:
            st.write("**Severity**")
            st.dataframe(
                df["severity"].astype(str).str.title()
                .value_counts(dropna=False).rename("Count").to_frame(),
                use_container_width=True
            )
        else:
            st.caption("No 'severity' column.")

    with right:
        if "urgency" in df.columns:
            st.write("**Urgency**")
            st.dataframe(
                df["urgency"].astype(str).str.title()
                .value_counts(dropna=False).rename("Count").to_frame(),
                use_container_width=True
            )
        else:
            st.caption("No 'urgency' column.")

    st.subheader("🗂️ Recent Cases")
    cols = [c for c in ["id","timestamp","subject","issue","status","severity","urgency"] if c in df.columns]
    if cols:
        df_sorted = df.sort_values(by="timestamp", ascending=False, na_position="last") if "timestamp" in df.columns else df
        st.dataframe(df_sorted[cols].head(20), use_container_width=True)
    else:
        st.caption("No preview columns available.")
