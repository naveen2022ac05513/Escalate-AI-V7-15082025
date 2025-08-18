# enhancement_dashboard.py
# EscalateAI — Enhancement Dashboard (standalone)

from __future__ import annotations
import os, sqlite3
from typing import Optional
import pandas as pd
import streamlit as st

DB_PATH = os.getenv("ESCALATEAI_DB_PATH") or os.getenv("DB_PATH", "escalations.db")

# ---------- helpers ----------

@st.cache_data(show_spinner=False)
def _table_exists(db_path: str, table: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            return conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is not None
    except Exception:
        return False

def _db_sig(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

@st.cache_data(show_spinner=True)
def load_escalations(db_path: str = DB_PATH, _sig: float | None = None) -> pd.DataFrame:
    _ = _sig if _sig is not None else _db_sig(db_path)
    if not os.path.exists(db_path) or not _table_exists(db_path, "escalations"):
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM escalations", conn)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for c in ["status","severity","urgency","criticality","sentiment","category",
              "likely_to_escalate","bu_code","bu_name","region"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "status" in df.columns:
        df["status"] = df["status"].str.strip().str.title()
    if "bu_code" in df.columns:
        df["bu_code"] = df["bu_code"].str.strip().str.upper()
    return df

def _kpi(label: str, value: int, help_text: Optional[str] = None):
    st.metric(label, f"{int(value):,}", help=help_text)

# ---------- main entry ----------

def show_enhancement_dashboard():
    st.header("🧠 Enhancements")

    df = load_escalations(DB_PATH, _db_sig(DB_PATH))
    if df.empty:
        st.info("No data available yet. Add some cases or upload an Excel to get started.")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1: _kpi("Total Cases", len(df))
    with col2: _kpi("Open", int((df.get("status") == "Open").sum()))
    with col3: _kpi("In Progress", int((df.get("status") == "In Progress").sum()))
    with col4: _kpi("Resolved", int((df.get("status") == "Resolved").sum()))

    st.divider()
    st.subheader("📦 Distribution Snapshots")

    # Try Altair; fallback to tables if unavailable
    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()

        def bar_with_labels(data: pd.DataFrame, x: str, y: str, title: str, height: int = 240):
            if data.empty:
                st.info(f"No data for {title}.")
                return
            base = alt.Chart(data).mark_bar().encode(
                x=alt.X(f"{x}:N", sort='-y', title=None),
                y=alt.Y(f"{y}:Q", title=None),
                color=alt.Color(f"{x}:N", legend=None),
                tooltip=[x, y],
            )
            labels = alt.Chart(data).mark_text(dy=-6).encode(text=f"{y}:Q")
            st.altair_chart((base + labels).properties(title=title, height=height), use_container_width=True)

        # BU distribution
        if "bu_code" in df.columns:
            bu = df["bu_code"].astype(str).value_counts(dropna=False).reset_index()
            bu.columns = ["BU", "Count"]
            bar_with_labels(bu, "BU", "Count", "BU Distribution")

        # Region distribution
        if "region" in df.columns:
            rg = df["region"].astype(str).str.title().value_counts(dropna=False).reset_index()
            rg.columns = ["Region", "Count"]
            bar_with_labels(rg, "Region", "Count", "Region Distribution")

        # Daily trend (all cases)
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            tdf = df.copy()
            tdf["date"] = pd.to_datetime(tdf["timestamp"], errors="coerce").dt.date
            vol = tdf.groupby("date").size().reset_index(name="Count")
            chart = alt.Chart(vol).mark_line(point=True).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Count:Q", title="Cases"),
                tooltip=["date:T", "Count:Q"],
            ).properties(title="Daily Case Volume", height=260)
            st.altair_chart(chart, use_container_width=True)

    except Exception:
        # Fallback tables
        colA, colB = st.columns(2)
        with colA:
            if "bu_code" in df.columns:
                st.write("**BU Distribution**")
                st.dataframe(df["bu_code"].value_counts(dropna=False).rename("Count").to_frame())
        with colB:
            if "region" in df.columns:
                st.write("**Region Distribution**")
                st.dataframe(df["region"].value_counts(dropna=False).rename("Count").to_frame())

        if "timestamp" in df.columns and df["timestamp"].notna().any():
            tdf = df.copy()
            tdf["date"] = pd.to_datetime(tdf["timestamp"], errors="coerce").dt.date
            st.line_chart(tdf.groupby("date").size())
