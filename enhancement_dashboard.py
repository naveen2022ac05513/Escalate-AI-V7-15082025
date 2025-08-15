import streamlit as st
from enhancements import fetch_escalations, train_model
from advanced_enhancements import generate_shap_plot, generate_pdf_report

def show_enhancement_dashboard():
    st.title("🧠 Enhancement Dashboard")
    st.markdown("Use this module to analyze likely escalations, generate SHAP plots, and download reports.")

    # Load data
    escalations = fetch_escalations()
    st.subheader("📋 Likely to Escalate Data")
    st.dataframe(escalations)

    # Model training
    st.subheader("⚙️ Train Model")
    if st.button("Train"):
        model = train_model()
        if model:
            st.success("Model trained successfully.")
        else:
            st.warning("Model training skipped — insufficient or invalid data.")

    # SHAP plot
    st.subheader("📊 SHAP Plot")
    model = train_model()
    if model:
        df = fetch_escalations()
        df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'likely_to_escalate'])
        X_sample = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
        generate_shap_plot(model, X_sample)

    # PDF report
    st.subheader("📄 Generate PDF Report")
    if st.button("Download Report"):
        pdf_bytes = generate_pdf_report()
        st.download_button("📥 Download PDF", data=pdf_bytes, file_name="enhancement_report.pdf")
