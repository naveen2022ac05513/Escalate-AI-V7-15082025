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

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DB_PATH = os.getenv("ESCALATEAI_DB_PATH") or os.getenv("DB_PATH", "escalations.db")


# ------------------------- Data Loading ------------------------- #
def _db_sig(db_path: str) -> float:
    try:
        return os.path.getmtime(db_path)
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _table_exists(db_path: str, table: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            return conn.execute(q, (table,)).fetchone() is not None
    except Exception:
        return False


@st.cache_data(show_spinner=True)
def load_escalations(db_path: str = DB_PATH, _sig: float | None = None) -> pd.DataFrame:
    """Cache is keyed on file mtime so it refreshes after DB writes."""
    _ = _sig if _sig is not None else _db_sig(db_path)
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


# ------------------------- Visuals ------------------------- #
def _render_sla_heatmap(df: pd.DataFrame):
    """Severity × Urgency count heatmap (pure matplotlib)."""
    if df is None or df.empty:
        st.info("No data for SLA heatmap.")
        return

    s = df.get("severity", pd.Series(dtype=str)).astype(str).str.title()
    u = df.get("urgency", pd.Series(dtype=str)).astype(str).str.title()
    pivot = pd.crosstab(s, u).sort_index()
    if pivot.empty:
        st.info("No data for SLA heatmap.")
        return

    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("SLA Heatmap — Count by Severity × Urgency")
    ax.set_xticks(np.arange(pivot.shape[1])); ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(pivot.values[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def _altair_or_table_for_counts(df_counts: pd.DataFrame, x: str, y: str, title: str, height: int = 240):
    """Render Altair bar chart if available, else show a table."""
    if df_counts.empty:
        st.info(f"No data for {title}.")
        return
    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()

        chart = alt.Chart(df_counts).mark_bar().encode(
            x=alt.X(f"{x}:N", sort='-y', title=x.replace("_", " ").title()),
            y=alt.Y(f"{y}:Q", title="Count"),
            tooltip=[x, y],
            color=alt.Color(f"{x}:N", legend=None),
        ).properties(height=height, title=title)

        labels = alt.Chart(df_counts).mark_text(dy=-6).encode(
            x=alt.X(f"{x}:N", sort='-y'), y=alt.Y(f"{y}:Q"), text=f"{y}:Q"
        )
        st.altair_chart(chart + labels, use_container_width=True)
    except Exception:
        st.dataframe(df_counts, use_container_width=True)


def _status_trend(df: pd.DataFrame):
    """Daily status mix line/area chart (Altair fallback to table)."""
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        st.info("No timestamps available for trends.")
        return

    tdf = df.copy()
    tdf["date"] = pd.to_datetime(tdf["timestamp"]).dt.date
    counts = (
        tdf.groupby(["date", "status"]).size().reset_index(name="count")
        .sort_values(["date", "status"])
    )

    if counts.empty:
        st.info("Insufficient data for status trends.")
        return

    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()
        chart = alt.Chart(counts).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title="Cases"),
            color=alt.Color("status:N", title="Status"),
            tooltip=["date:T", "status:N", "count:Q"],
        ).properties(height=240, title="Status Trend by Day")
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(
            counts.pivot(index="date", columns="status", values="count").fillna(0),
            use_container_width=True
        )


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

    st.divider()

    # ---------------- 2×2 Grid ---------------- #
    a, b = st.columns(2)
    with a:
        st.subheader("🔥 SLA Heatmap")
        try:
            _render_sla_heatmap(df)
        except Exception as e:
            st.error(f"❌ SLA Heatmap failed to render: {type(e).__name__}: {e}")

    with b:
        st.subheader("📈 Status Trend")
        _status_trend(df)

    c, d = st.columns(2)
    with c:
        st.subheader("🧱 Severity Distribution")
        sev_counts = _value_counts_frame(df, "severity")
        _altair_or_table_for_counts(sev_counts.rename(columns={"count": "Count"}), "severity", "Count", "Severity")

    with d:
        st.subheader("⏱️ Urgency Distribution")
        urg_counts = _value_counts_frame(df, "urgency")
        _altair_or_table_for_counts(urg_counts.rename(columns={"count": "Count"}), "urgency", "Count", "Urgency")

    # ---------------- Optional Model Debug ---------------- #
    st.divider()
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

    fi = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()
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
