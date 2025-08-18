# enhancement_dashboard.py
# Robust Enhancement Dashboard for EscalateAI
# - No hard dependency on advanced_enhancements / enhancements modules
# - Safe fallbacks for charts, SHAP, PDF export
# - Advanced Analytics laid out as 2×2 grid
# - BU/Region visuals moved to a dedicated tab
# - Does not alter your main app's search bar or capsule layout

from __future__ import annotations

import sys
import os
import sqlite3
import warnings
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st

# Charts & ML
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------- Utilities -------------------------

def _get_main_attr(name: str):
    """Fetch a callable/attr from the __main__ module if present."""
    main_mod = sys.modules.get("__main__")
    return getattr(main_mod, name, None) if main_mod else None


def _read_from_sqlite(db_path: str = "escalations.db") -> Optional[pd.DataFrame]:
    """SQLite fallback if the main app/session doesn’t expose a dataframe."""
    if not os.path.exists(db_path):
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM escalations", conn)
        return df if not df.empty else None
    except Exception:
        return None


def _get_dataframe() -> Optional[pd.DataFrame]:
    """
    Try, in order:
      1) st.session_state['escalations_df']
      2) __main__.get_escalations_df()
      3) __main__.load_all_escalations()
      4) sqlite: ./escalations.db, table 'escalations'
    """
    # 1) Session cache
    df = st.session_state.get("escalations_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()

    # 2) main.get_escalations_df()
    getter = _get_main_attr("get_escalations_df")
    if callable(getter):
        try:
            df = getter()
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()
        except Exception:
            pass

    # 3) main.load_all_escalations()
    loader = _get_main_attr("load_all_escalations")
    if callable(loader):
        try:
            df = loader()
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()
        except Exception:
            pass

    # 4) sqlite fallback
    return _read_from_sqlite()


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _compute_age_hours(df: pd.DataFrame,
                       created_col: str = "timestamp",
                       closed_col: str = "closed_at") -> pd.DataFrame:
    """Compute age in hours from created to now/closed_at."""
    if created_col not in df.columns:
        return df

    now = pd.Timestamp.now(tz=None)
    df[created_col] = pd.to_datetime(df[created_col], errors="coerce")

    if closed_col in df.columns:
        df[closed_col] = pd.to_datetime(df[closed_col], errors="coerce")
        end_time = df[closed_col].fillna(now)
    else:
        end_time = pd.Series([now] * len(df), index=df.index)

    age = (end_time - df[created_col]).dt.total_seconds() / 3600.0
    df["age_hours"] = age.round(2)
    return df


def _coalesce_cols(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first present column from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _likely_to_escalate_series(dff: pd.DataFrame) -> Optional[pd.Series]:
    """
    Derive a boolean series for 'likely to escalate' from common columns:
    - predicted_escalation / escalation_flag / prediction / likely_to_escalate
    Accepts 1/0, True/False, 'yes'/'no', 'high'/'low', etc.
    """
    col = _coalesce_cols(dff, [
        "predicted_escalation", "likely_to_escalate",
        "escalation_flag", "prediction", "predicted_flag"
    ])
    if not col:
        return None

    s = dff[col]
    if pd.api.types.is_bool_dtype(s):
        return s

    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).fillna(0.0) > 0.5

    # string-ish
    vals = s.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "yes", "y", "high", "escalate", "likely", "positive"}
    return vals.isin(truthy)


def _sentiment_label_series(dff: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return a categorical sentiment series. Tries:
      - sentiment_label / sentiment_class / sentiment
      - or bucket numeric sentiment_score into Neg/Neu/Pos using +/-0.05
    """
    col = _coalesce_cols(dff, ["sentiment_label", "sentiment_class", "sentiment"])
    if col:
        return dff[col].astype(str)

    # Try numeric score bucketing
    score_col = _coalesce_cols(dff, ["sentiment_score", "vader_compound", "compound"])
    if score_col and pd.api.types.is_numeric_dtype(dff[score_col]):
        score = pd.to_numeric(dff[score_col], errors="coerce").fillna(0.0)
        lab = pd.Series(np.where(score > 0.05, "Positive",
                        np.where(score < -0.05, "Negative", "Neutral")), index=dff.index)
        return lab
    return None


def _safe_feature_importance(model, X: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    Returns (importances_df, fig). If SHAP is present, tries a SHAP summary bar.
    Otherwise falls back to permutation importance. fig can be None.
    """
    feature_names = list(X.columns)

    # Try SHAP
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X, check_additivity=False)

        # Handle binary/multi-class
        if isinstance(shap_values, list) and len(shap_values) > 1:
            vals = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            vals = np.abs(sv).mean(axis=0)

        imp_df = pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values(
            "importance", ascending=False
        )
        fig = plt.figure()
        shap.summary_plot(shap_values, X, show=False, plot_type="bar", max_display=15)
        return imp_df, fig

    except Exception:
        # Fallback: permutation importance (uses in-sample predictions)
        try:
            res = permutation_importance(model, X, model.predict(X), n_repeats=8, random_state=42)
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": res.importances_mean
            }).sort_values("importance", ascending=False)

            fig = plt.figure()
            top = imp_df.head(15)
            plt.barh(top["feature"][::-1], top["importance"][::-1])
            plt.title("Feature Importance (Permutation)")
            plt.xlabel("Importance")
            return imp_df, fig
        except Exception:
            return pd.DataFrame(columns=["feature", "importance"]), None


def _make_pdf_report(df: pd.DataFrame, metrics: dict) -> Optional[str]:
    """Create a tiny PDF report if reportlab is installed; else return None."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        path = "enhancement_report.pdf"
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4

        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h - 50, "EscalateAI – Enhancement Analytics Report")

        c.setFont("Helvetica", 11)
        y = h - 90
        for k, v in metrics.items():
            c.drawString(40, y, f"{k}: {v}")
            y -= 16

        c.showPage()
        c.save()
        return path
    except Exception:
        return None


# ------------------------- Main Dashboard -------------------------

def show_enhancement_dashboard() -> None:
    """
    Enhancement Dashboard:
      - Loads data from session/main/sqlite
      - Renders analytics in a 2×2 grid
      - BU/Region charts moved to 'BU & Region Trends' tab
      - Skips gracefully when columns/models are missing
    """
    st.subheader("🔧 Enhancement Dashboard")

    df = _get_dataframe()
    if df is None or df.empty:
        st.info("No data found to render Enhancement Dashboard.")
        return

    # Normalize timestamps & compute age
    df = _ensure_datetime(df, "timestamp")
    df = _ensure_datetime(df, "closed_at")
    df = _compute_age_hours(df)

    # Local filters (keeps your global search bar untouched)
    with st.expander("Filters (local to Enhancement Dashboard)"):
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            date_min_default = df["timestamp"].min().date()
            date_max_default = df["timestamp"].max().date()
        else:
            today = pd.Timestamp.today().date()
            date_min_default = today
            date_max_default = today

        date_min = st.date_input("From date", value=date_min_default)
        date_max = st.date_input("To date", value=date_max_default)

        status_pick = "All"
        if "status" in df.columns:
            statuses = ["All"] + sorted([str(s) for s in df["status"].dropna().unique()])
            status_pick = st.selectbox("Status", statuses, index=0)

    # Apply filters
    dff = df.copy()
    if "timestamp" in dff.columns and dff["timestamp"].notna().any():
        mask = (dff["timestamp"].dt.date >= date_min) & (dff["timestamp"].dt.date <= date_max)
        dff = dff.loc[mask].copy()

    if status_pick != "All" and "status" in dff.columns:
        dff = dff[dff["status"].astype(str) == status_pick]

    # Quick KPIs (kept minimal; no “capsules” added under your main AI summary)
    total_cases = int(len(dff))
    open_cnt = int((dff.get("status", pd.Series()).astype(str) == "Open").sum()) if "status" in dff.columns else 0
    inprog_cnt = int((dff.get("status", pd.Series()).astype(str) == "In Progress").sum()) if "status" in dff.columns else 0
    resolved_cnt = int((dff.get("status", pd.Series()).astype(str) == "Resolved").sum()) if "status" in dff.columns else 0
    escalate_series = _likely_to_escalate_series(dff)
    likely_cnt = int(escalate_series.sum()) if escalate_series is not None else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total (filtered)", total_cases)
    k2.metric("Open", open_cnt)
    k3.metric("In Progress", inprog_cnt)
    k4.metric("Resolved", resolved_cnt)
    k5.metric("Likely to Escalate", likely_cnt)

    tabs = st.tabs(["Advanced Analytics", "BU & Region Trends"])

    # ---------------- Advanced Analytics (2×2 grid) ----------------
    with tabs[0]:
        st.caption("All charts arranged in **2 columns × 2 rows** as requested.")

        top_row = st.columns(2)
        bot_row = st.columns(2)

        # Chart 1: Cases per Day
        with top_row[0]:
            if "timestamp" in dff.columns and dff["timestamp"].notna().any():
                tmp = dff.copy()
                tmp["day"] = tmp["timestamp"].dt.date
                series = tmp.groupby("day").size()
                fig = plt.figure()
                plt.plot(series.index, series.values, marker="o")
                plt.title("Cases per Day")
                plt.xlabel("Day")
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No 'timestamp' column to draw 'Cases per Day'.")

        # Chart 2: Status Distribution
        with top_row[1]:
            if "status" in dff.columns:
                series = dff["status"].astype(str).fillna("Unknown").value_counts()
                fig = plt.figure()
                plt.bar(series.index, series.values)
                plt.title("Status Distribution")
                plt.xticks(rotation=20)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No 'status' column to draw 'Status Distribution'.")

        # Chart 3: Sentiment Distribution
        with bot_row[0]:
            sent_series = _sentiment_label_series(dff)
            if sent_series is not None:
                series = sent_series.fillna("Unknown").value_counts()
                fig = plt.figure()
                plt.bar(series.index, series.values)
                plt.title("Sentiment Distribution")
                plt.xticks(rotation=20)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No sentiment labels/scores found to plot.")

        # Chart 4: SLA Breaches per Day
        with bot_row[1]:
            if "sla_breached" in dff.columns and "timestamp" in dff.columns:
                tmp = dff.copy()
                tmp["day"] = tmp["timestamp"].dt.date
                breached_mask = (tmp["sla_breached"].fillna(0).astype(float) > 0.5)
                series = tmp[breached_mask].groupby("day").size() if breached_mask.any() else pd.Series(dtype=int)
                fig = plt.figure()
                if not series.empty:
                    plt.plot(series.index, series.values, marker="o")
                plt.title("SLA Breaches per Day")
                plt.xlabel("Day")
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Need 'sla_breached' and 'timestamp' to draw SLA breaches.")

        st.divider()

        # Optional: Model insights (feature importance)
        model_fn = _get_main_attr("get_current_model") or _get_main_attr("train_model")
        build_features = _get_main_attr("build_features")

        if callable(model_fn) and callable(build_features):
            try:
                out = build_features(dff)
                X = out[0] if isinstance(out, (tuple, list)) else out
                model = model_fn() if callable(model_fn) else None

                if X is not None and hasattr(model, "predict"):
                    st.markdown("**Model Feature Importance**")
                    imp_df, fig = _safe_feature_importance(model, X)
                    if fig:
                        st.pyplot(fig, clear_figure=True)
                    if not imp_df.empty:
                        st.dataframe(imp_df.reset_index(drop=True).head(20), use_container_width=True)
            except Exception as e:
                st.info(f"Model insights skipped: {type(e).__name__}: {e}")

        # Quick PDF export (optional)
        metrics = {
            "Total (filtered)": total_cases,
            "Open": open_cnt,
            "In Progress": inprog_cnt,
            "Resolved": resolved_cnt,
            "Likely to Escalate": likely_cnt,
            "Avg Age (hrs)": round(float(np.nanmean(dff.get("age_hours", pd.Series(dtype=float)))), 2)
                             if "age_hours" in dff.columns and not dff["age_hours"].empty else 0.0
        }
        if st.button("📄 Export Enhancement PDF"):
            path = _make_pdf_report(dff, metrics)
            if path and os.path.exists(path):
                st.success(f"Report created: {path}")
                with open(path, "rb") as fp:
                    st.download_button("Download PDF", data=fp.read(), file_name=path, mime="application/pdf")
            else:
                st.warning("PDF generation unavailable (install reportlab to enable).")

    # ---------------- BU & Region Trends ----------------
    with tabs[1]:
        # Column name flexibility
        bu_col = _coalesce_cols(dff, ["business_unit", "bu_code", "bu"])
        rg_col = _coalesce_cols(dff, ["region", "geo", "area"])

        c1, c2 = st.columns(2)

        with c1:
            if bu_col:
                series = dff[bu_col].astype(str).fillna("Unknown").value_counts()
                fig = plt.figure()
                plt.bar(series.index, series.values)
                plt.title("Cases by Business Unit")
                plt.xticks(rotation=20)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Business Unit column not found (expected one of: business_unit, bu_code, bu).")

        with c2:
            if rg_col:
                series = dff[rg_col].astype(str).fillna("Unknown").value_counts()
                fig = plt.figure()
                plt.bar(series.index, series.values)
                plt.title("Cases by Region")
                plt.xticks(rotation=20)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Region column not found (expected one of: region, geo, area).")

        st.divider()

        # Monthly trends
        if "timestamp" in dff.columns and dff["timestamp"].notna().any():
            tdf = dff.copy()
            tdf["month"] = pd.to_datetime(tdf["timestamp"]).dt.to_period("M").dt.to_timestamp()

            cols = st.columns(2)
            with cols[0]:
                if bu_col:
                    t_bu = tdf.groupby(["month", bu_col]).size().reset_index(name="Count")
                    fig = plt.figure()
                    for label, sub in t_bu.groupby(bu_col):
                        plt.plot(sub["month"], sub["Count"], marker="o", label=str(label))
                    plt.title("Monthly Trend by BU")
                    plt.xlabel("Month")
                    plt.ylabel("Cases")
                    if not t_bu.empty:
                        plt.legend(loc="best", fontsize=8)
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("BU column not found for monthly trend.")

            with cols[1]:
                if rg_col:
                    t_rg = tdf.groupby(["month", rg_col]).size().reset_index(name="Count")
                    fig = plt.figure()
                    for label, sub in t_rg.groupby(rg_col):
                        plt.plot(sub["month"], sub["Count"], marker="o", label=str(label))
                    plt.title("Monthly Trend by Region")
                    plt.xlabel("Month")
                    plt.ylabel("Cases")
                    if not t_rg.empty:
                        plt.legend(loc="best", fontsize=8)
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Region column not found for monthly trend.")
        else:
            st.info("No usable 'timestamp' to draw monthly trends.")

        st.divider()

        # BU × Status and Region × Status matrices
        id_like = "id" if "id" in dff.columns else (dff.columns[0] if len(dff.columns) else "Count")

        if bu_col and "status" in dff.columns:
            pivot = pd.pivot_table(dff, index=bu_col, columns="status", values=id_like,
                                   aggfunc="count", fill_value=0)
            st.write("**BU × Status**")
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Need BU and Status to show BU × Status.")

        if rg_col and "status" in dff.columns:
            pivot = pd.pivot_table(dff, index=rg_col, columns="status", values=id_like,
                                   aggfunc="count", fill_value=0)
            st.write("**Region × Status**")
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Need Region and Status to show Region × Status.")
