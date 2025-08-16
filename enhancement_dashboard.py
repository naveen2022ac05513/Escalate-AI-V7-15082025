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

# ✅ Ensure pandas is defined at import time
import pandas as pd
import numpy as np
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
def load_escalations(db_path: str = DB_PATH, _sig: float = None) -> pd.DataFrame:
    """Cache is keyed on file mtime so it refreshes after DB writes."""
    _sig = _sig if _sig is not None else _db_sig(db_path)
    if not os.path.exists(db_path) or not _table_exists(db_path, "escalations"):
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM escalations", conn)

    # Normalize
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Cast only non-numeric columns; keep labels for ML later
    for c in ["status", "severity", "urgency", "criticality", "sentiment",
              "category", "bu_code", "bu_name", "region"]:
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


def _count_status(df: pd.DataFrame, status_value: str) -> int:
    if "status" not in df.columns:
        return 0
    s = df["status"].astype(str).str.strip().str.title()
    return int((s == status_value).sum())


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
    Safe, self-contained Streamlit dashboard. Optional deps (altair, sklearn, shap)
    are imported lazily and handled gracefully if missing.
    """
    st.header("🧠 Enhancements")

    # --- Load data
    df = load_escalations(DB_PATH, _db_sig(DB_PATH))
    if df.empty:
        st.info("No data available yet. Add some cases or upload an Excel to get started.")
        return

    # --- KPIs
    st.subheader("📌 Quick KPIs")
    c1, c2, c3, c4 = st.columns(4)
    with c1: _kpi("Total Cases", len(df))
    with c2: _kpi("Open", _count_status(df, "Open"))
    with c3: _kpi("In Progress", _count_status(df, "In Progress"))
    with c4: _kpi("Resolved", _count_status(df, "Resolved"))

    # --- Charts (Altair is optional)
    try:
        import altair as alt  # lazy import
        has_alt = True
    except Exception:
        has_alt = False

    st.subheader("📦 Distribution Snapshots")
    if has_alt:
        d1, d2 = st.columns(2)
        with d1:
            bu_counts = _value_counts_frame(df, "bu_code", top_n=20)
            _plot_labelled_bar(bu_counts, "bu_code", "count", "BU Distribution (labelled)", alt)
        with d2:
            reg_counts = _value_counts_frame(df, "region", top_n=20)
            _plot_labelled_bar(reg_counts, "region", "count", "Region Distribution (labelled)", alt)
    else:
        st.caption("Install Altair for charts: `pip install altair`")
        st.dataframe(_value_counts_frame(df, "bu_code"), use_container_width=True)
        st.dataframe(_value_counts_frame(df, "region"), use_container_width=True)

    # --- Daily Trend
    st.subheader("📈 Daily Trend")
    if "timestamp" in df.columns and df["timestamp"].notna().any() and has_alt:
        day = df["timestamp"].dt.date
        trend = (
            pd.DataFrame({"date": day})
            .groupby("date", dropna=True)
            .size()
            .reset_index(name="count")
            .sort_values("date")
        )
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title="Cases"),
            tooltip=["date:T", "count:Q"]
        ).properties(height=280, title="Total Cases per Day")
        st.altair_chart(chart, use_container_width=True)
    elif "timestamp" in df.columns and df["timestamp"].notna().any():
        st.dataframe(
            pd.DataFrame({"date": df["timestamp"].dt.date}).value_counts().reset_index(name="count").sort_values("date"),
            use_container_width=True
        )
    else:
        st.info("No timestamps to compute daily trend.")

    # --- Status Mix
    st.subheader("🧩 Status Mix by Day")
    if {"timestamp", "status"} <= set(df.columns) and df["timestamp"].notna().any() and has_alt:
        tmp = df.copy()
        tmp["date"] = tmp["timestamp"].dt.date
        mix = tmp.groupby(["date", "status"]).size().reset_index(name="count")
        chart = alt.Chart(mix).mark_bar().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", stack="zero", title="Cases"),
            color=alt.Color("status:N", title="Status"),
            tooltip=["date:T", "status:N", "count:Q"],
        ).properties(height=280, title="Stacked Status per Day")
        st.altair_chart(chart, use_container_width=True)
    elif {"timestamp", "status"} <= set(df.columns) and df["timestamp"].notna().any():
        st.dataframe(
            df.assign(date=df["timestamp"].dt.date)
              .groupby(["date", "status"]).size().reset_index(name="count")
              .pivot(index="date", columns="status", values="count").fillna(0),
            use_container_width=True
        )
    else:
        st.info("Insufficient data for status mix.")

    # --- Model Debug (optional sklearn)
    st.subheader("🧪 Model Debug (experimental)")
    needed = {"sentiment", "urgency", "severity", "criticality", "likely_to_escalate"}
    if not needed.issubset(df.columns):
        st.caption("Missing columns for model debug. Need: " + ", ".join(sorted(needed)))
        return

    df_md = df.dropna(subset=list(needed)).copy()
    # accept 0/1/true/false/yes/no
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

    try:
        from sklearn.ensemble import RandomForestClassifier  # lazy import
        has_skl = True
    except Exception:
        has_skl = False

    if (not has_skl) or len(df_md) < 30:
        st.caption("Need scikit-learn and at least ~30 labelled rows to show feature importances.")
        return

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y.values)

    fi = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    if has_alt:
        import altair as alt  # already imported above, but safe
        chart = alt.Chart(fi).mark_bar().encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort='-x', title="Feature"),
            tooltip=["feature:N", alt.Tooltip("importance:Q", format=".3f")],
            color=alt.Color("feature:N", legend=None),
        ).properties(height=360, title="Top Feature Importances")
        labels = alt.Chart(fi).mark_text(align='left', dx=4).encode(
            x="importance:Q",
            y=alt.Y("feature:N", sort='-x'),
            text=alt.Text("importance:Q", format=".3f"),
        )
        st.altair_chart(chart + labels, use_container_width=True)
    else:
        st.dataframe(fi, use_container_width=True)

    # --- Optional SHAP
    try:
        import shap  # lazy import
        has_shap = True
    except Exception:
        has_shap = False

    st.markdown("**📊 SHAP Plot**")
    if has_shap:
        try:
            explainer = shap.TreeExplainer(rf)
            # keep it light for performance
            Xs = X.sample(min(len(X), 200), random_state=42)
            shap_vals = explainer.shap_values(Xs)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_vals[1] if isinstance(shap_vals, list) else shap_vals, Xs, show=False)
            st.pyplot(bbox_inches="tight", clear_figure=True)
        except Exception as e:
            st.caption(f"SHAP could not render ({type(e).__name__}).")
    else:
        st.caption("Install SHAP to view plots: `pip install shap`")


# ------------------------- Chart helper ------------------------- #
def _plot_labelled_bar(df_counts: pd.DataFrame, x: str, y: str, title: str, alt_module):
    """Altair bar with count labels on top."""
    if df_counts.empty:
        st.info(f"No data for {title}.")
        return
    base = alt_module.Chart(df_counts).mark_bar().encode(
        x=alt_module.X(f"{x}:N", sort='-y', title=x.replace("_", " ").title()),
        y=alt_module.Y(f"{y}:Q", title="Count"),
        color=alt_module.Color(f"{x}:N", legend=None),
        tooltip=[x, y],
    ).properties(height=280, title=title)
    labels = alt_module.Chart(df_counts).mark_text(dy=-6).encode(
        x=alt_module.X(f"{x}:N", sort='-y'),
        y=alt_module.Y(f"{y}:Q"),
        text=f"{y}:Q"
    )
    st.altair_chart(base + labels, use_container_width=True)
