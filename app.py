"""
app.py
------
Interactive Streamlit Web Application: Customer Conversion Prediction
Uses Scikit-learn trained ML pipeline to predict prospective customer conversion.
"""

import json
import os
from typing import Optional, Tuple

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash

# Load environment configuration
load_dotenv()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip()
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@conversion.ai").strip()
RAW_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
if not ADMIN_PASSWORD_HASH:
    ADMIN_PASSWORD_HASH = generate_password_hash(RAW_ADMIN_PASSWORD, method="pbkdf2:sha256")

# Configure page layout and meta information
st.set_page_config(
    page_title="Customer Conversion Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "customer_conversion_model.joblib")
RAW_DATA_PATH = os.path.join(BASE_DIR, "customer_enquiries.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "customer_enquiries_cleaned.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_JSON_PATH = os.path.join(OUTPUTS_DIR, "model_evaluation_metrics.json")


# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #94A3B8;
        font-size: 1.05rem;
    }
    
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .result-card-converted {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .result-card-not-converted {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
    }
    
    .badge-converted {
        background-color: #10B981;
        color: white;
    }
    
    .badge-not-converted {
        background-color: #EF4444;
        color: white;
    }
    
    .badge-label {
        background-color: #334155;
        color: #F8FAFC;
        margin-left: 0.5rem;
    }
    
    .recommendation-box {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-top: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# CACHED MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained machine learning model...")
def load_trained_model():
    """Loads the pre-trained Scikit-learn Pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None


@st.cache_data
def load_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict]]:
    """Loads raw dataset, cleaned dataset, and evaluation metrics."""
    raw_df = pd.read_csv(RAW_DATA_PATH) if os.path.exists(RAW_DATA_PATH) else None
    clean_df = pd.read_csv(CLEAN_DATA_PATH) if os.path.exists(CLEAN_DATA_PATH) else None
    metrics = None
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r") as f:
            metrics = json.load(f)
    return raw_df, clean_df, metrics


# Helper for normalizing text input matching pipeline expectations
def normalize_text_input(val: str) -> str:
    mapping = {
        "website": "Website",
        "referral": "Referral",
        "social media": "Social Media",
        "advertisement": "Advertisement",
        "walk-in": "Walk-in",
        "basic": "Basic",
        "standard": "Standard",
        "premium": "Premium",
        "enterprise": "Enterprise",
    }
    clean = str(val).strip()
    return mapping.get(clean.lower(), clean.title())


# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🎯 Customer Conversion Prediction</h1>
        <p>AI-driven Lead Scoring & Prospective Customer Conversion Estimation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with business context and settings
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/commercial.png", width=70)
    st.title("About This System")
    st.markdown(
        """
        **Business Objective:**
        Analyze historical prospective customer enquiries to estimate whether a new prospect will convert into an active paying customer.
        
        - **Label 1 (`Converted`)**: Prospect became a paying client.
        - **Label 0 (`Not Converted`)**: Prospect dropped off.
        
        **Benefits:**
        - Prioritize sales team follow-up efforts.
        - Maximize lead-to-deal velocity.
        - Identify at-risk prospects early.
        """
    )
    st.markdown("---")
# -----------------------------------------------------------------------------
# ADMIN AUTHENTICATION GATE
# -----------------------------------------------------------------------------
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

with st.sidebar:
    if st.session_state["admin_authenticated"]:
        st.markdown(f"**👤 Active Admin:** `{ADMIN_USERNAME}`")
        if st.button("🚪 Logout Admin", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.rerun()
        st.markdown("---")

if not st.session_state["admin_authenticated"]:
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("### 🔐 Admin Authentication Required")
        st.caption("Enter your administrator credentials to access the Customer Conversion Prediction System.")
        
        with st.form("admin_login_form"):
            login_user = st.text_input("Username or Admin Email", placeholder="admin or admin@conversion.ai")
            show_pwd = st.checkbox("Show password", value=False)
            login_password = st.text_input("Password", type="text" if show_pwd else "password", placeholder="••••••••••••")
            login_submit = st.form_submit_button("Sign In to Dashboard", type="primary", use_container_width=True)

        if login_submit:
            if not login_user.strip() or not login_password:
                st.error("❌ Please enter both username/email and password.")
            else:
                u_clean = login_user.strip().lower()
                user_match = (u_clean == ADMIN_USERNAME.lower()) or (u_clean == ADMIN_EMAIL.lower())
                pass_match = check_password_hash(ADMIN_PASSWORD_HASH, login_password)
                if user_match and pass_match:
                    st.session_state["admin_authenticated"] = True
                    st.success("✅ Authenticated successfully! Loading dashboard...")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1rem; font-size: 0.8rem; color: #475569;">
                💡 <b>Default Demo Credentials:</b> User: <code>admin</code> | Password: <code>admin123</code>
            </div>
            <div style="text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 2rem;">
                ⚡ Powered by Machine Learning
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# Load model and datasets once authenticated
pipeline = load_trained_model()
raw_df, clean_df, metrics_data = load_datasets()

if pipeline is None:
    st.error("⚠️ Trained model not found at `models/customer_conversion_model.joblib`. Please run `python train_model.py` first.")
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training model and generating artifacts..."):
            import subprocess
            subprocess.run(["python", "train_model.py"], check=True)
            st.success("Model trained successfully! Please refresh.")
            st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
tab_predict, tab_evaluation, tab_eda, tab_data = st.tabs(
    ["🎯 Predict Conversion", "📊 Model Evaluation & Metrics", "📈 Exploratory Data Analysis", "📁 Dataset & Data Quality"]
)


# =============================================================================
# TAB 1: PREDICT CONVERSION
# =============================================================================
with tab_predict:
    st.subheader("Estimate Conversion Likelihood for a Prospective Customer")
    st.write(
        "Fill in the five enquiry attributes below to evaluate conversion probability and receive targeted sales follow-up recommendations."
    )

    # Preset scenario helper buttons
    st.markdown("**⚡ Quick Test Presets:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    preset_choice = None
    with p_col1:
        if st.button("🌟 High-Intent Referral"):
            preset_choice = "referral"
    with p_col2:
        if st.button("🏢 Enterprise Prospect"):
            preset_choice = "enterprise"
    with p_col3:
        if st.button("❄️ Slow Social Media Lead"):
            preset_choice = "cold_social"
    with p_col4:
        if st.button("🌐 Average Web Enquiry"):
            preset_choice = "average_web"

    # Default values based on preset
    default_source = "Website"
    default_service = "Standard"
    default_budget = 5000.0
    default_response_time = 4.0
    default_follow_ups = 3

    if preset_choice == "referral":
        default_source = "Referral"
        default_service = "Enterprise"
        default_budget = 28000.0
        default_response_time = 1.5
        default_follow_ups = 4
    elif preset_choice == "enterprise":
        default_source = "Walk-in"
        default_service = "Enterprise"
        default_budget = 35000.0
        default_response_time = 2.0
        default_follow_ups = 5
    elif preset_choice == "cold_social":
        default_source = "Social Media"
        default_service = "Basic"
        default_budget = 1100.0
        default_response_time = 42.0
        default_follow_ups = 0
    elif preset_choice == "average_web":
        default_source = "Website"
        default_service = "Standard"
        default_budget = 4800.0
        default_response_time = 12.0
        default_follow_ups = 2

    # Prediction Input Form
    with st.form("enquiry_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            source = st.selectbox(
                "1. Enquiry Source (`source`)",
                options=["Website", "Referral", "Social Media", "Advertisement", "Walk-in"],
                index=["Website", "Referral", "Social Media", "Advertisement", "Walk-in"].index(default_source),
                help="Channel through which the customer enquiry arrived.",
            )

            service_type = st.selectbox(
                "2. Service Tier (`service_type`)",
                options=["Basic", "Standard", "Premium", "Enterprise"],
                index=["Basic", "Standard", "Premium", "Enterprise"].index(default_service),
                help="Tier of service or package the prospective customer requested.",
            )

            budget = st.number_input(
                "3. Customer Budget ($) (`budget`)",
                min_value=0.0,
                max_value=150000.0,
                value=float(default_budget),
                step=100.0,
                format="%.2f",
                help="Estimated or stated budget for the requested service.",
            )

        with col2:
            response_time_hours = st.number_input(
                "4. Response Time in Hours (`response_time_hours`)",
                min_value=0.0,
                max_value=168.0,
                value=float(default_response_time),
                step=0.5,
                format="%.1f",
                help="Number of hours elapsed between the enquiry receipt and the company's first response.",
            )

            previous_interactions_follow_up_count = st.number_input(
                "5. Follow-up / Interaction Count (`previous_interactions_follow_up_count`)",
                min_value=0,
                max_value=30,
                value=int(default_follow_ups),
                step=1,
                help="Number of touchpoints, calls, emails, or meetings already completed.",
            )

            st.markdown(
                """
                <div style="background-color: #F1F5F9; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1.5rem; font-size: 0.85rem; color: #475569;">
                    ℹ️ <b>Validation Active:</b> All inputs are checked against domain limits prior to pipeline inference.
                </div>
                """,
                unsafe_allow_html=True,
            )

        submit_btn = st.form_submit_button("🔮 Predict Conversion", use_container_width=True, type="primary")

    # Perform Validation and Inference upon submission
    if submit_btn:
        validation_errors = []

        if budget < 0:
            validation_errors.append("Budget must be a non-negative number.")
        if response_time_hours < 0:
            validation_errors.append("Response time cannot be negative.")
        if previous_interactions_follow_up_count < 0:
            validation_errors.append("Follow-up interaction count cannot be negative.")

        if validation_errors:
            for err in validation_errors:
                st.error(f"❌ Input Validation Error: {err}")
        else:
            # Construct DataFrame with the exact feature column names used during training
            input_data = pd.DataFrame(
                [
                    {
                        "source": normalize_text_input(source),
                        "service_type": normalize_text_input(service_type),
                        "budget": float(budget),
                        "response_time_hours": float(response_time_hours),
                        "previous_interactions_follow_up_count": int(previous_interactions_follow_up_count),
                    }
                ]
            )

            # Predict using the cached trained Scikit-learn Pipeline
            try:
                pred_label = int(pipeline.predict(input_data)[0])
                pred_proba = float(pipeline.predict_proba(input_data)[0][1]) * 100.0
                pred_class = "Converted" if pred_label == 1 else "Not Converted"

                # Display Results in a Rich Visual Card
                if pred_label == 1:
                    card_class = "result-card-converted"
                    badge_class = "badge-converted"
                    status_icon = "✅"
                    badge_text = "CONVERTED"
                    rec_priority = "High Priority"
                    rec_action = (
                        "Assign lead to an Account Executive immediately. "
                        "Because this enquiry shows strong conversion signals, conduct an executive demo within 24 hours."
                    )
                else:
                    card_class = "result-card-not-converted"
                    badge_class = "badge-not-converted"
                    status_icon = "⚠️"
                    badge_text = "NOT CONVERTED"
                    rec_priority = "Nurture / Follow-up Needed"
                    rec_action = (
                        "Engage with an automated educational email campaign. "
                        "Consider scheduling a low-pressure discovery check-in or offering an introductory case study."
                    )

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <span class="badge-pill {badge_class}">{status_icon} {badge_text}</span>
                                <span class="badge-pill badge-label">Label: {pred_label}</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #1E293B;">
                                Estimated Conversion Probability: <span style="color: {'#047857' if pred_label == 1 else '#B91C1C'}">{pred_proba:.1f}%</span>
                            </div>
                        </div>
                        <div style="margin-top: 1rem;">
                            <div style="background-color: #E2E8F0; border-radius: 9999px; height: 12px; overflow: hidden; width: 100%;">
                                <div style="width: {min(max(pred_proba, 2.0), 100.0):.1f}%; background-color: {'#10B981' if pred_proba >= 50 else ('#F59E0B' if pred_proba >= 30 else '#EF4444')}; height: 100%;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748B; margin-top: 4px;">
                                <span>0% (Cold)</span>
                                <span>50% (Threshold)</span>
                                <span>100% (High Intent)</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Business Recommendation Callout
                st.markdown(
                    f"""
                    <div class="recommendation-box">
                        <strong style="color: #1D4ED8; font-size: 1.05rem;">📋 Strategic Recommendation ({rec_priority}):</strong>
                        <p style="margin: 0.3rem 0 0 0; color: #1E293B; font-size: 0.95rem;">{rec_action}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Mandatory Disclaimer
                st.caption(
                    "⚠️ **Disclaimer:** The conversion probability percentage shown above is a statistical model estimate "
                    "derived from historical enquiry patterns. It is an operational prioritization tool and not a guaranteed business outcome."
                )

                # Features breakdown table
                with st.expander("🔍 View Preprocessed Enquiry Features Sent to Pipeline"):
                    st.dataframe(input_data, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")


# =============================================================================
# TAB 2: MODEL EVALUATION & METRICS
# =============================================================================
with tab_evaluation:
    st.subheader("Model Evaluation & Performance Benchmarking")
    st.write(
        "Two candidate binary classification models were trained and evaluated on a held-out test split (20% sample, stratified by target class)."
    )

    if metrics_data:
        m_lr = metrics_data["models"]["Logistic Regression"]
        m_rf = metrics_data["models"]["Random Forest Classifier"]

        # Comparison summary metrics cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Best Model", metrics_data["best_model"])
        c2.metric("Best Model Accuracy", f"{m_rf['accuracy']*100:.1f}%")
        c3.metric("Best Model F1-Score", f"{m_rf['f1_score']:.2f}")
        c4.metric("Best Model ROC-AUC", f"{m_rf['roc_auc']:.2f}")

        st.markdown("---")

        # Side-by-side metric comparison table
        st.markdown("### 📊 Metrics Comparison Table")
        comparison_df = pd.DataFrame(
            {
                "Evaluation Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC Score"],
                "Logistic Regression": [
                    f"{m_lr['accuracy']:.4f}",
                    f"{m_lr['precision']:.4f}",
                    f"{m_lr['recall']:.4f}",
                    f"{m_lr['f1_score']:.4f}",
                    f"{m_lr['roc_auc']:.4f}",
                ],
                "Random Forest Classifier": [
                    f"{m_rf['accuracy']:.4f}",
                    f"{m_rf['precision']:.4f}",
                    f"{m_rf['recall']:.4f}",
                    f"{m_rf['f1_score']:.4f}",
                    f"{m_rf['roc_auc']:.4f}",
                ],
            }
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Visualizations
    st.markdown("### 📈 Evaluation Plots")
    ev_col1, ev_col2 = st.columns(2)

    cm_img = os.path.join(OUTPUTS_DIR, "model_confusion_matrix.png")
    roc_img = os.path.join(OUTPUTS_DIR, "model_roc_curve.png")
    comp_img = os.path.join(OUTPUTS_DIR, "model_comparison.png")

    if os.path.exists(cm_img):
        with ev_col1:
            st.image(cm_img, caption="Confusion Matrices: Logistic Regression vs Random Forest", use_container_width=True)

    if os.path.exists(roc_img):
        with ev_col2:
            st.image(roc_img, caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

    if os.path.exists(comp_img):
        st.image(comp_img, caption="Comprehensive Model Performance Comparison Across All Metrics", use_container_width=True)

    # Plain language metric explanation
    st.markdown("### 💡 Plain-Language Guide to Evaluation Metrics")
    st.markdown(
        """
        - **Accuracy**: The proportion of all predictions that were correct. *Caution:* If 70% of enquiries do not convert, a model that simply predicts 0 for everyone would achieve 70% accuracy while failing to identify any sales opportunities.
        - **Precision**: When the model predicts a customer will convert, how often is it right? High precision ensures the sales team doesn't waste time on cold leads.
        - **Recall (Sensitivity)**: Out of all prospective customers who actually converted, how many did the model identify? High recall ensures we don't miss lucrative conversion opportunities.
        - **F1-Score**: The harmonic mean of precision and recall. It is the gold standard for imbalanced classification because it balances both false alarms and missed conversions.
        - **ROC-AUC**: Evaluates the model's ability to rank prospects across all decision thresholds. A score of 0.5 is random guessing; 1.0 is a perfect classifier.
        """
    )
    st.info(
        "📌 **Synthetic Dataset Note:** The models were trained on a 100-sample synthetic dataset designed for teaching and architecture demonstration. The numerical metrics are illustrative."
    )


# =============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA) Insights")
    st.write(
        "Visual analysis of the 100 historical enquiry records reveals meaningful relationships between enquiry characteristics and conversion rates."
    )

    eda1 = os.path.join(OUTPUTS_DIR, "eda_conversion_counts.png")
    eda2 = os.path.join(OUTPUTS_DIR, "eda_conversion_by_source.png")
    eda3 = os.path.join(OUTPUTS_DIR, "eda_conversion_by_service_type.png")
    eda4 = os.path.join(OUTPUTS_DIR, "eda_budget_distribution.png")
    eda5 = os.path.join(OUTPUTS_DIR, "eda_response_time_vs_conversion.png")
    eda6 = os.path.join(OUTPUTS_DIR, "eda_missing_values_before_after.png")

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(eda1):
            st.image(eda1, caption="Figure 1: Overall Enquiry Conversion Distribution", use_container_width=True)
            st.caption("📌 **Insight:** 33% baseline conversion rate demonstrates realistic class imbalance.")
    with col_b:
        if os.path.exists(eda2):
            st.image(eda2, caption="Figure 2: Conversion Rate by Enquiry Source", use_container_width=True)
            st.caption("📌 **Insight:** Referrals and Walk-ins exhibit significantly higher conversion propensity than cold ad clicks.")

    col_c, col_d = st.columns(2)
    with col_c:
        if os.path.exists(eda3):
            st.image(eda3, caption="Figure 3: Conversion Rate by Service Tier", use_container_width=True)
            st.caption("📌 **Insight:** Enterprise and Premium enquiries convert at higher rates due to high customer intent.")
    with col_d:
        if os.path.exists(eda5):
            st.image(eda5, caption="Figure 4: Response Time (Hours) vs Conversion", use_container_width=True)
            st.caption("📌 **Insight:** Prospects contacted rapidly (< 5-10 hours) are markedly more likely to convert.")

    col_e, col_f = st.columns(2)
    with col_e:
        if os.path.exists(eda4):
            st.image(eda4, caption="Figure 5: Customer Budget Distribution", use_container_width=True)
            st.caption("📌 **Insight:** Higher budgets correlate positively with conversion, particularly for enterprise contracts.")
    with col_f:
        if os.path.exists(eda6):
            st.image(eda6, caption="Figure 6: Missing Values Before vs After Data Cleaning", use_container_width=True)
            st.caption("📌 **Insight:** Median and mode imputation successfully resolved all missing values.")


# =============================================================================
# TAB 4: DATASET & DATA QUALITY EXPLORER
# =============================================================================
with tab_data:
    st.subheader("Data Quality Inspection: Raw vs Cleaned Dataset")
    st.write(
        "The project intentionally introduces realistic data quality issues in `customer_enquiries.csv` "
        "and resolves them in `customer_enquiries_cleaned.csv`."
    )

    st.markdown("### 🛠️ Data Quality Issues Introduced & Resolved")
    st.markdown(
        """
        | Category | Raw Issue in `customer_enquiries.csv` | Cleaning Applied in Pipeline |
        |---|---|---|
        | **Categorical Whitespace/Casing** | Entries like `'website'`, `'Website '`, `'WEBSITE'`, `'premium'`, `'PREMIUM'` | Whitespace stripped, unified with case-insensitive normalization. |
        | **Missing Values** | Missing entries in `source` (3), `service_type` (3), `budget` (4), `response_time_hours` (4) | Imputed with most frequent category (mode) and numerical median. |
        | **Invalid Numeric Values** | Negative response times (`-4.5h`) and negative follow-up counts (`-2`) | Replaced with `NaN` during cleaning and properly imputed. |
        | **Identifier Column** | `enquiry_id` present as an index | Excluded from modeling feature matrix $X$ to prevent spurious correlation. |
        """
    )

    data_view_mode = st.radio("Choose dataset to view:", ["Cleaned Dataset (100 rows)", "Raw Dataset with Quality Issues (100 rows)"], horizontal=True)

    if data_view_mode == "Cleaned Dataset (100 rows)" and clean_df is not None:
        st.dataframe(clean_df, use_container_width=True)
        st.caption(f"Cleaned dataset shape: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns. Missing values: {clean_df.isnull().sum().sum()}")
    elif raw_df is not None:
        st.dataframe(raw_df, use_container_width=True)
        st.caption(f"Raw dataset shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns. Missing values: {raw_df.isnull().sum().sum()}")
