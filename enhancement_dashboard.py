# enhancement_dashboard.py
# -------------------------------------------------------------------
# EscalateAI — Enhancement Dashboard (standalone, safe to import)
# -------------------------------------------------------------------
# Usage from main app:
#   from enhancement_dashboard import show_enhancement_dashboard
#   show_enhancement_dashboard()
# -------------------------------------------------------------------

from __future__ import annotations

import os
import sqlite3
from typing import Optional

# ✅ Ensure pandas is present at import time (fixes NameError: pd)
import pandas as pd
import streamlit as st

DB_PATH = os.getenv("ESCALATEAI_DB_PATH") or os.getenv("DB_PATH", "escalations.db")


# ------------------------- Data Loading ------------------------- #
@st.cache_data(show_spinner=False)
def _table_exists(db_path: str, table: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            return conn.execute(q, (table,)).fetchone() is not None
    except Exception:
        return False


def _db_sig(db_path: str) -> float:
    try:
        return os.path.getmtime(db_path)
    except Exception:
        return 0.0


@st.cache_data(show_spinner=True)
def load_escalations(db_path: str = DB_PATH, _sig: float | None = None) -> pd.DataFrame:
    """Cache is keyed on file mtime so it refreshes after DB writes."""
    _sig = _sig if _sig is not None else _db_sig(db_path)
    if not os.path.exists(db_path) or not _table_exists(db_path, "escalations"):
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM escalations", conn)

    # Normalize
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
    """
    Safe, self-contained Streamlit dashboard.
    Optional deps (altair, sklearn, shap) are imported lazily and handled gracefully if missing.
    """
    st.header("🧠 Enhancements")

    df = load_escalations(DB_PATH, _db_sig(DB_PATH))
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

    # Try Altair; if missing, show tables instead
    try:
        import altair as alt  # lazy import

        def _labelled_bar(df_counts: pd.DataFrame, x: str, y: str, title: str):
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

        
    # ---------------- 2×2 Distribution Snapshots ---------------- #
    st.subheader("📦 Distribution Snapshots")
    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()
        def _border(ch, height): 
            return (ch.properties(height=height)
                      .configure_view(stroke='#CBD5E1', strokeWidth=1)
                      .configure_axis(grid=True, domain=True))
        # BU distribution
        bu_counts = (df["bu_code"].astype(str).fillna("")
                        .replace({"nan": ""})
                        .value_counts(dropna=False)
                        .reset_index()
                        .rename(columns={"index":"BU","bu_code":"Count"}))
        # Region distribution
        region_counts = (df["region"].astype(str).fillna("")
                            .replace({"nan": ""})
                            .value_counts(dropna=False)
                            .reset_index()
                            .rename(columns={"index":"Region","region":"Count"}))
        # Monthly trends
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            tdf = df.copy()
            tdf["month"] = pd.to_datetime(tdf["timestamp"]).dt.to_period("M").dt.to_timestamp()
            t_bu = (tdf.groupby(["month","bu_code"]).size().reset_index(name="Count"))
            t_rg = (tdf.groupby(["month","region"]).size().reset_index(name="Count"))
        else:
            t_bu = pd.DataFrame({"month": [], "bu_code": [], "Count": []})
            t_rg = pd.DataFrame({"month": [], "region": [], "Count": []})

        # charts
        bu_bar = alt.Chart(bu_counts).mark_bar().encode(
            x=alt.X("BU:N", sort="-y"), y=alt.Y("Count:Q"), tooltip=["BU","Count"]
        ).properties(title="BU Distribution")
        rg_bar = alt.Chart(region_counts).mark_bar().encode(
            x=alt.X("Region:N", sort="-y"), y=alt.Y("Count:Q"), tooltip=["Region","Count"]
        ).properties(title="Region Distribution")
        bu_line = alt.Chart(t_bu).mark_line(point=True).encode(
            x=alt.X("month:T", title="Month"), y=alt.Y("Count:Q", title="Cases"),
            color=alt.Color("bu_code:N", title="BU"), tooltip=["month:T","bu_code:N","Count:Q"]
        ).properties(title="Monthly Trend by BU")
        rg_line = alt.Chart(t_rg).mark_line(point=True).encode(
            x=alt.X("month:T", title="Month"), y=alt.Y("Count:Q", title="Cases"),
            color=alt.Color("region:N", title="Region"), tooltip=["month:T","region:N","Count:Q"]
        ).properties(title="Monthly Trend by Region")

        c1, c2 = st.columns(2)
        with c1: st.altair_chart(_border(bu_bar, 220), use_container_width=True)
        with c2: st.altair_chart(_border(rg_bar, 220), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3: st.altair_chart(_border(bu_line, 240), use_container_width=True)
        with c4: st.altair_chart(_border(rg_line, 240), use_container_width=True)

    except Exception:
        st.caption("Install Altair for charts: `pip install altair`")
        col1, col2 = st.columns(2)
        with col1: st.dataframe(df[['bu_code']].value_counts().reset_index(name='Count'), use_container_width=True)
        with col2: st.dataframe(df[['region']].value_counts().reset_index(name='Count'), use_container_width=True)

            st.dataframe(
                df.assign(date=df["timestamp"].dt.date)
                  .groupby(["date", "status"]).size().reset_index(name="count")
                  .pivot(index="date", columns="status", values="count").fillna(0),
                use_container_width=True
            )
    else:
        st.info("Insufficient data for status mix.")

    # ---------------- Model Debug (optional) ---------------- #
    st.subheader("🧪 Model Debug (experimental)")
    needed = {"sentiment", "urgency", "severity", "criticality", "likely_to_escalate"}
    if not needed.issubset(df.columns):
        st.caption("Missing columns for model debug. Need: " + ", ".join(sorted(needed)))
        return

    df_md = df.dropna(subset=list(needed)).copy()
    # Normalize labels (accept yes/no/1/0/true/false)
    y = (
        df_md["likely_to_escalate"].astype(str).str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
    )
    df_md = df_md[~y.isna()]
    y = y[~y.isna()]
    if df_md.empty or y.nunique() < 2:
        st.caption("Not enough labelled variation in 'likely_to_escalate' to train a model.")
        return

    X = pd.get_dummies(df_md[["sentiment", "urgency", "severity", "criticality"]])

    # Lazy import sklearn; bail gracefully if missing
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        st.caption("Install scikit-learn (`pip install scikit-learn`) to view feature importances.")
        return

    if len(df_md) < 30:
        st.caption("Need at least ~30 labelled rows to show feature importances.")
        return

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y.values)

    # Feature importances
    fi = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    try:
        import altair as alt
        fi_chart = alt.Chart(fi).mark_bar().encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort='-x', title="Feature"),
            tooltip=["feature:N", alt.Tooltip("importance:Q", format=".3f")],
            color=alt.Color("feature:N", legend=None),
        ).properties(height=360, title="Top Feature Importances")
        labels = alt.Chart(fi).mark_text(align='left', dx=4).encode(
            x="importance:Q", y=alt.Y("feature:N", sort='-x'),
            text=alt.Text("importance:Q", format=".3f"),
        )
        st.altair_chart(fi_chart + labels, use_container_width=True)
    except Exception:
        st.dataframe(fi, use_container_width=True)

    # Optional SHAP
    st.markdown("**📊 SHAP Plot**")
    try:
        import shap
        try:
            explainer = shap.TreeExplainer(rf)
            Xs = X.sample(min(len(X), 200), random_state=42)
            shap_vals = explainer.shap_values(Xs)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_vals[1] if isinstance(shap_vals, list) else shap_vals, Xs, show=False)
            st.pyplot(bbox_inches="tight", clear_figure=True)
        except Exception:
            st.caption("SHAP is installed but failed to render for this dataset.")
    except Exception:
        st.caption("Install SHAP to view plots: `pip install shap`")
