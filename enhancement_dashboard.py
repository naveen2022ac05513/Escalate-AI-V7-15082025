# enhancement_dashboard.py
# -------------------------------------------------------------------
# EscalateAI — Enhancement Dashboard (standalone, safe to import)
# -------------------------------------------------------------------
# Usage from main app:
#   from enhancement_dashboard import show_enhancement_dashboard
#   show_enhancement_dashboard()
#
# Reads from SQLite (escalations.db) directly; no imports from main
# Robust to low-data / missing columns
# KPIs: Total / Open / In Progress / Resolved
# Labelled bar charts for BU & Region distributions
# Daily trend and status mix
# Optional “Model Debug” (RandomForest) + optional SHAP (if installed)
# -------------------------------------------------------------------

import os
import sqlite3
from typing import Optional

# ✅ ensure pandas is defined early
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Optional ML (safe fallbacks if not installed)
try:
    from sklearn.ensemble import RandomForestClassifier
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

DB_PATH = os.getenv("DB_PATH", "escalations.db")

# ------------------------- Data Loading ------------------------- #
@st.cache_data(show_spinner=False)
def _table_exists(db_path: str, table: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            return conn.execute(q, (table,)).fetchone() is not None
    except Exception:
        return False


@st.cache_data(show_spinner=True)
def load_escalations(db_path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(db_path) or not _table_exists(db_path, "escalations"):
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM escalations", conn)

    # Normalize common columns
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for c in [
        "status", "severity", "urgency", "criticality", "sentiment",
        "category", "likely_to_escalate", "bu_code", "bu_name", "region"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if "status" in df.columns:
        df["status"] = df["status"].str.strip().str.title()

    if "bu_code" in df.columns:
        df["bu_code"] = df["bu_code"].str.strip().str.upper()

    return df

# ------------------------- Small utils ------------------------- #
def _kpi(label: str, value: int, help_text: Optional[str] = None):
    st.metric(label, f"{value:,}", help=help_text)


def _labelled_bar(df_counts: pd.DataFrame, x: str, y: str, title: str):
    """Altair bar with count labels on top."""
    if df_counts.empty:
        st.info(f"No data for {title}.")
        return

    base = alt.Chart(df_counts).mark_bar().encode(
        x=alt.X(f"{x}:N", sort='-y', title=x.replace("_", " ").title()),
        y=alt.Y(f"{y}:Q", title="Count"),
        color=alt.Color(f"{x}:N", legend=None),
        tooltip=[x, y],
    ).properties(height=280, title=title)

    labels = alt.Chart(df_counts).mark_text(dy=-6).encode(
        x=alt.X(f"{x}:N", sort='-y'),
        y=alt.Y(f"{y}:Q"),
        text=f"{y}:Q"
    )

    st.altair_chart(base + labels, use_container_width=True)


def _value_counts_frame(df: pd.DataFrame, col: str, top_n: Optional[int] = None) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame({col: [], "count": []})
    vc = (
        df[col].fillna("").replace({"nan": ""})
        .value_counts(dropna=False).reset_index()
        .rename(columns={"index": col, col: "count"})
    )
    if top_n is not None:
        vc = vc.head(top_n)
    return vc

# --------------------- Enhancement Dashboard -------------------- #
def show_enhancement_dashboard():
    st.header("🧠 Enhancements")

    df = load_escalations()
    if df.empty:
        st.info("No data available yet. Add some cases or upload an Excel to get started.")
        return

    # ---------------- KPIs ---------------- #
    st.subheader("📌 Quick KPIs")
    c1, c2, c3, c4 = st.columns(4)
    with c1: _kpi("Total Cases", len(df))
    with c2: _kpi("Open", int((df.get("status") == "Open").sum()))
    with c3: _kpi("In Progress", int((df.get("status") == "In Progress").sum()))
    with c4: _kpi("Resolved", int((df.get("status") == "Resolved").sum()))

    # ---------------- Distributions ---------------- #
    st.subheader("📦 Distribution Snapshots")
    d1, d2 = st.columns(2)
    with d1: _labelled_bar(_value_counts_frame(df, "bu_code"), "bu_code", "count", "BU Distribution")
    with d2: _labelled_bar(_value_counts_frame(df, "region"), "region", "count", "Region Distribution")

    # ---------------- Daily Trends ---------------- #
    st.subheader("📈 Daily Trend")
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        day = df["timestamp"].dt.date
        trend = pd.DataFrame({"date": day}).value_counts().reset_index(name="count").sort_values("date")
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x="date:T", y="count:Q", tooltip=["date:T", "count:Q"]
        ).properties(height=280, title="Total Cases per Day")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No timestamps to compute daily trend.")

    # ---------------- Status Mix ---------------- #
    st.subheader("🧩 Status Mix by Day")
    if {"timestamp", "status"} <= set(df.columns) and df["timestamp"].notna().any():
        tmp = df.copy()
        tmp["date"] = tmp["timestamp"].dt.date
        mix = tmp.groupby(["date", "status"]).size().reset_index(name="count")
        chart = alt.Chart(mix).mark_bar().encode(
            x="date:T", y="count:Q", color="status:N",
            tooltip=["date:T", "status:N", "count:Q"]
        ).properties(height=280, title="Stacked Status per Day")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Insufficient data for status mix.")

    # ---------------- Model Debug ---------------- #
    st.subheader("🧪 Model Debug (experimental)")
    needed = {"sentiment", "urgency", "severity", "criticality", "likely_to_escalate"}
    if not needed.issubset(df.columns):
        st.caption("Missing columns for model debug. Need: " + ", ".join(sorted(needed)))
        return

    df_md = df.dropna(subset=list(needed)).copy()
    if df_md.empty or df_md["likely_to_escalate"].nunique() < 2:
        st.caption("Not enough variation in 'likely_to_escalate' to train.")
        return

    X = pd.get_dummies(df_md[["sentiment", "urgency", "severity", "criticality"]])
    y = df_md["likely_to_escalate"].astype(str).str.lower().eq("yes").astype(int)

    if (not _HAS_SK) or len(df_md) < 30:
        st.caption("Need scikit-learn and ~30 labelled rows to show importances.")
        return

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False).head(20)
    chart = alt.Chart(fi).mark_bar().encode(
        x="importance:Q", y=alt.Y("feature:N", sort='-x'),
        tooltip=["feature:N", "importance:Q"], color="feature:N"
    ).properties(height=360, title="Top Feature Importances")
    labels = alt.Chart(fi).mark_text(align='left', dx=4).encode(
        x="importance:Q", y=alt.Y("feature:N", sort='-x'),
        text=alt.Text("importance:Q", format=".3f"),
    )
    st.altair_chart(chart + labels, use_container_width=True)

    st.markdown("**📊 SHAP Plot** (if `shap` is installed)")
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(X.sample(min(len(X), 200), random_state=42))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_vals[1] if isinstance(shap_vals, list) else shap_vals, X, show=False)
            st.pyplot(bbox_inches="tight", clear_figure=True)
        except Exception as e:
            st.caption(f"SHAP failed ({type(e).__name__}).")
    else:
        st.caption("Install `shap` to view SHAP plots: `pip install shap`")
