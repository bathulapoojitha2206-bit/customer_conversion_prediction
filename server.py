"""
server.py
---------
Flask Application for Customer Conversion Prediction System
Features:
- Secure Session-Based Admin Authentication (Werkzeug password hashing, sessions, route protection)
- Protected Admin Dashboard
- Real-time Machine Learning Inference using the serialized Scikit-learn Pipeline
- RESTful APIs for Predictions, Customer Insights, and KPI Metrics
"""

import json
import os
import sys
from datetime import timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

# -----------------------------------------------------------------------------
# DIRECTORY & ENVIRONMENT CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for .env in current directory or parent directory
env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(env_path):
    env_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

# Admin Credentials & Security Configuration
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip()
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@conversion.ai").strip()
RAW_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")

# If no pre-computed hash provided, generate a secure cryptographic hash in memory
if not ADMIN_PASSWORD_HASH:
    ADMIN_PASSWORD_HASH = generate_password_hash(RAW_ADMIN_PASSWORD, method="pbkdf2:sha256")

SECRET_KEY = os.getenv("SECRET_KEY", "sixthsense_conversion_prediction_secret_key_2026_secure")
SESSION_LIFETIME = int(os.getenv("SESSION_LIFETIME_SECONDS", "3600"))

# Paths for ML Model and Data
MODEL_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "models", "customer_conversion_model.joblib"),
    os.path.join(os.path.dirname(BASE_DIR), "models", "customer_conversion_model.joblib"),
    os.path.join(BASE_DIR, "customer_conversion_model.joblib"),
]

DATA_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "customer_enquiries_cleaned.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "customer_enquiries_cleaned.csv"),
    os.path.join(BASE_DIR, "customer_enquiries.csv"),
]

METRICS_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "outputs", "model_evaluation_metrics.json"),
    os.path.join(os.path.dirname(BASE_DIR), "outputs", "model_evaluation_metrics.json"),
]

# -----------------------------------------------------------------------------
# FLASK APPLICATION SETUP
# -----------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

app.secret_key = SECRET_KEY
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(seconds=SESSION_LIFETIME)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# -----------------------------------------------------------------------------
# ML MODEL & DATA STORE
# -----------------------------------------------------------------------------
_ml_pipeline = None
_customer_dataset: List[Dict[str, Any]] = []
_metrics_data: Optional[Dict[str, Any]] = None


def get_model_path() -> Optional[str]:
    for p in MODEL_CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None


def get_data_path() -> Optional[str]:
    for p in DATA_CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None


def get_metrics_path() -> Optional[str]:
    for p in METRICS_CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None


def load_model():
    global _ml_pipeline
    if _ml_pipeline is None:
        path = get_model_path()
        if path and os.path.exists(path):
            try:
                _ml_pipeline = joblib.load(path)
                print(f"[ML Engine] Pretrained model loaded successfully from: {path}")
            except Exception as e:
                print(f"[ML Engine] Error loading model from {path}: {e}")
        else:
            print("[ML Engine] Warning: Model file not found in candidate paths.")
    return _ml_pipeline


def initialize_customer_data():
    global _customer_dataset, _metrics_data
    data_path = get_data_path()
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            # Normalize casing and fill missing for clean display
            records = []
            for idx, row in df.iterrows():
                enquiry_id = str(row.get("enquiry_id", f"ENQ{idx+1:03d}"))
                source = normalize_text_input(str(row.get("source", "Website")))
                service = normalize_text_input(str(row.get("service_type", "Standard")))
                budget = float(row.get("budget", 5000.0) if pd.notna(row.get("budget")) else 5000.0)
                resp_time = float(row.get("response_time_hours", 12.0) if pd.notna(row.get("response_time_hours")) else 12.0)
                follow_ups = int(row.get("previous_interactions_follow_up_count", 2) if pd.notna(row.get("previous_interactions_follow_up_count")) else 2)
                converted = int(row.get("converted", 0))

                records.append({
                    "enquiry_id": enquiry_id,
                    "source": source,
                    "service_type": service,
                    "budget": round(budget, 2),
                    "response_time_hours": round(resp_time, 1),
                    "previous_interactions_follow_up_count": follow_ups,
                    "converted": converted,
                    "status_label": "Converted" if converted == 1 else "Not Converted",
                    "probability": 85.0 if converted == 1 else 20.0,  # Historical label
                    "source_type": "Historical Record",
                })
            _customer_dataset = records
            print(f"[Data Engine] Loaded {len(_customer_dataset)} customer records from {data_path}")
        except Exception as e:
            print(f"[Data Engine] Error loading data from {data_path}: {e}")

    # Load metrics if available
    metrics_path = get_metrics_path()
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                _metrics_data = json.load(f)
        except Exception as e:
            print(f"[Data Engine] Error reading metrics JSON: {e}")


# Normalization helper matching the Scikit-learn Pipeline's expectations
def normalize_text_input(val: str) -> str:
    mapping = {
        "website": "Website",
        "referral": "Referral",
        "social media": "Social Media",
        "advertisement": "Advertisement",
        "walk-in": "Walk-in",
        "walk in": "Walk-in",
        "basic": "Basic",
        "standard": "Standard",
        "premium": "Premium",
        "enterprise": "Enterprise",
    }
    clean = str(val).strip()
    return mapping.get(clean.lower(), clean.title())


# Initialize at startup
load_model()
initialize_customer_data()


# -----------------------------------------------------------------------------
# AUTHENTICATION & ACCESS CONTROL HELPERS
# -----------------------------------------------------------------------------
def is_authenticated() -> bool:
    """Checks if the active session has valid admin authentication."""
    return session.get("is_admin") is True


def login_required(f):
    """Decorator to enforce admin authentication on protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            if request.is_json or request.path.startswith("/api/"):
                return (
                    jsonify({
                        "success": False,
                        "error": "Unauthorized",
                        "message": "Admin authentication required. Please log in.",
                    }),
                    401,
                )
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)

    return decorated_function


# -----------------------------------------------------------------------------
# APPLICATION ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Root redirector: sends authenticated admins to dashboard, others to login."""
    if is_authenticated():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Admin Login Page & Authentication Handler.
    Supports both traditional Form POST and Modern AJAX JSON submission.
    """
    if is_authenticated():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        # Extract payload from JSON or Form Data
        if request.is_json:
            data = request.get_json() or {}
            username = str(data.get("username", "")).strip()
            password = str(data.get("password", ""))
            remember_me = bool(data.get("remember_me", False))
        else:
            username = str(request.form.get("username", "")).strip()
            password = str(request.form.get("password", ""))
            remember_me = bool(request.form.get("remember_me"))

        # Validation: Check for empty credentials
        if not username or not password:
            error_msg = "Please enter both username/email and password."
            if request.is_json:
                return jsonify({"success": False, "message": error_msg}), 400
            return render_template("login.html", error=error_msg, username=username)

        # Verification: Check username/email match and cryptographic password hash
        user_matches = (username.lower() == ADMIN_USERNAME.lower()) or (username.lower() == ADMIN_EMAIL.lower())
        password_matches = check_password_hash(ADMIN_PASSWORD_HASH, password)

        if not user_matches or not password_matches:
            error_msg = "Invalid username or password."
            if request.is_json:
                return jsonify({"success": False, "message": error_msg}), 401
            return render_template("login.html", error=error_msg, username=username)

        # Authentication Successful: Establish secure session
        session.clear()
        session["is_admin"] = True
        session["username"] = ADMIN_USERNAME
        session["email"] = ADMIN_EMAIL
        session.permanent = remember_me

        next_url = request.args.get("next") or url_for("dashboard")

        if request.is_json:
            return jsonify({
                "success": True,
                "message": "Login successful. Redirecting...",
                "redirect_url": next_url,
                "user": {"username": ADMIN_USERNAME, "email": ADMIN_EMAIL},
            })

        return redirect(next_url)

    # GET Request: Render Login Page
    error = request.args.get("error")
    info = None
    if request.args.get("logout") == "1":
        info = "You have been successfully logged out."
    elif request.args.get("next"):
        info = "Please log in to access the Admin Dashboard."

    return render_template("login.html", error=error, info=info)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    """Terminates admin session and redirects to login."""
    session.clear()
    if request.is_json:
        return jsonify({"success": True, "message": "Logged out successfully.", "redirect_url": url_for("login")})
    return redirect(url_for("login", logout="1"))


@app.route("/dashboard")
@login_required
def dashboard():
    """Protected Admin Dashboard Route."""
    model = load_model()
    model_loaded = model is not None
    model_name = "Random Forest Classifier (Scikit-learn Pipeline)"
    if _metrics_data and "best_model" in _metrics_data:
        model_name = _metrics_data["best_model"]

    # Calculate initial stats
    total = len(_customer_dataset)
    converted = sum(1 for c in _customer_dataset if c.get("converted") == 1)
    non_converted = total - converted
    rate = round((converted / total * 100), 1) if total > 0 else 0.0

    return render_template(
        "dashboard.html",
        username=session.get("username", ADMIN_USERNAME),
        email=session.get("email", ADMIN_EMAIL),
        model_loaded=model_loaded,
        model_name=model_name,
        stats={
            "total": total,
            "converted": converted,
            "non_converted": non_converted,
            "rate": rate,
        },
    )


# -----------------------------------------------------------------------------
# RESTFUL API ENDPOINTS (PROTECTED)
# -----------------------------------------------------------------------------
@app.route("/api/stats", methods=["GET"])
@login_required
def get_stats():
    """Returns dynamic KPI metrics and distribution summaries."""
    total = len(_customer_dataset)
    converted = sum(1 for c in _customer_dataset if c.get("converted") == 1)
    non_converted = total - converted
    rate = round((converted / total * 100), 1) if total > 0 else 0.0

    # Source breakdown
    sources = ["Website", "Referral", "Social Media", "Advertisement", "Walk-in"]
    source_stats = {}
    for s in sources:
        matching = [c for c in _customer_dataset if c.get("source") == s]
        conv_count = sum(1 for c in matching if c.get("converted") == 1)
        tot_count = len(matching)
        source_stats[s] = {
            "total": tot_count,
            "converted": conv_count,
            "rate": round((conv_count / tot_count * 100), 1) if tot_count > 0 else 0.0,
        }

    # Service tier breakdown
    services = ["Basic", "Standard", "Premium", "Enterprise"]
    service_stats = {}
    for st_name in services:
        matching = [c for c in _customer_dataset if c.get("service_type") == st_name]
        conv_count = sum(1 for c in matching if c.get("converted") == 1)
        tot_count = len(matching)
        service_stats[st_name] = {
            "total": tot_count,
            "converted": conv_count,
            "rate": round((conv_count / tot_count * 100), 1) if tot_count > 0 else 0.0,
        }

    return jsonify({
        "success": True,
        "stats": {
            "total": total,
            "converted": converted,
            "non_converted": non_converted,
            "rate": rate,
            "by_source": source_stats,
            "by_service": service_stats,
        },
    })


@app.route("/api/customers", methods=["GET"])
@login_required
def get_customers():
    """Returns the customer list for table presentation."""
    # Reverse list so newest predictions appear at the top
    return jsonify({
        "success": True,
        "count": len(_customer_dataset),
        "customers": list(reversed(_customer_dataset)),
    })


@app.route("/api/predict", methods=["POST"])
@login_required
def predict_conversion():
    """
    ML Prediction Endpoint:
    Receives customer details, validates inputs, executes Scikit-learn Pipeline inference,
    returns conversion classification, probability percentage, and business recommendations.
    Appends newly predicted lead to in-memory customer dataset.
    """
    model = load_model()
    if model is None:
        return (
            jsonify({
                "success": False,
                "error": "Model Unavailable",
                "message": "The trained model pipeline could not be loaded. Please ensure customer_conversion_model.joblib exists.",
            }),
            500,
        )

    data = request.get_json() or {}

    # Extract inputs with validation
    source_raw = str(data.get("source", "Website")).strip()
    service_raw = str(data.get("service_type", "Standard")).strip()

    valid_sources = ["Website", "Referral", "Social Media", "Advertisement", "Walk-in"]
    valid_services = ["Basic", "Standard", "Premium", "Enterprise"]

    source = normalize_text_input(source_raw)
    service_type = normalize_text_input(service_raw)

    if source not in valid_sources:
        return (
            jsonify({
                "success": False,
                "error": "Validation Error",
                "message": f"Invalid source '{source_raw}'. Allowed values: {', '.join(valid_sources)}",
            }),
            400,
        )

    if service_type not in valid_services:
        return (
            jsonify({
                "success": False,
                "error": "Validation Error",
                "message": f"Invalid service tier '{service_raw}'. Allowed values: {', '.join(valid_services)}",
            }),
            400,
        )

    try:
        budget = float(data.get("budget", 0.0))
        if budget < 0:
            return jsonify({"success": False, "error": "Validation Error", "message": "Customer budget must be non-negative."}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Validation Error", "message": "Invalid numeric budget."}), 400

    try:
        response_time = float(data.get("response_time_hours", 0.0))
        if response_time < 0:
            return jsonify({"success": False, "error": "Validation Error", "message": "Response time cannot be negative."}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Validation Error", "message": "Invalid response time."}), 400

    try:
        follow_ups = int(data.get("previous_interactions_follow_up_count", 0))
        if follow_ups < 0:
            return jsonify({"success": False, "error": "Validation Error", "message": "Follow-up count cannot be negative."}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Validation Error", "message": "Invalid follow-up count."}), 400

    # Construct feature DataFrame using identical column schema
    input_df = pd.DataFrame([
        {
            "source": source,
            "service_type": service_type,
            "budget": float(budget),
            "response_time_hours": float(response_time),
            "previous_interactions_follow_up_count": int(follow_ups),
        }
    ])

    try:
        pred_label = int(model.predict(input_df)[0])
        pred_proba = float(model.predict_proba(input_df)[0][1]) * 100.0
    except Exception as e:
        return (
            jsonify({
                "success": False,
                "error": "Inference Error",
                "message": f"Error running ML inference: {str(e)}",
            }),
            500,
        )

    # Class & Strategic Recommendations
    if pred_label == 1:
        class_label = "Converted"
        badge_type = "success"
        priority = "High Priority"
        recommendation = (
            "Assign lead to an Account Executive immediately. "
            "Because this enquiry shows strong conversion signals, conduct an executive demo within 24 hours."
        )
    else:
        class_label = "Not Converted"
        badge_type = "danger"
        priority = "Nurture / Follow-up Needed"
        recommendation = (
            "Engage with an automated educational email campaign. "
            "Consider scheduling a low-pressure discovery check-in or offering an introductory case study."
        )

    # Confidence category
    if pred_proba >= 65.0:
        intent = "High Intent"
    elif pred_proba >= 40.0:
        intent = "Moderate Intent"
    else:
        intent = "Low Intent"

    # Add to customer dataset
    new_id = f"ENQ{len(_customer_dataset) + 1:03d}"
    record = {
        "enquiry_id": new_id,
        "source": source,
        "service_type": service_type,
        "budget": round(budget, 2),
        "response_time_hours": round(response_time, 1),
        "previous_interactions_follow_up_count": follow_ups,
        "converted": pred_label,
        "status_label": class_label,
        "probability": round(pred_proba, 1),
        "source_type": "Live Admin Inference",
    }
    _customer_dataset.append(record)

    return jsonify({
        "success": True,
        "prediction": pred_label,
        "class_label": class_label,
        "probability": round(pred_proba, 1),
        "badge_type": badge_type,
        "intent": intent,
        "priority": priority,
        "recommendation": recommendation,
        "enquiry_id": new_id,
        "inputs": {
            "source": source,
            "service_type": service_type,
            "budget": budget,
            "response_time_hours": response_time,
            "previous_interactions_follow_up_count": follow_ups,
        },
    })


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    print(f"\n" + "=" * 70)
    print("CUSTOMER CONVERSION PREDICTION SYSTEM - ADMIN PORTAL")
    print("=" * 70)
    print(f"-> Local Server: http://{host}:{port}")
    print(f"-> Admin Username: {ADMIN_USERNAME}")
    print(f"-> Admin Email:    {ADMIN_EMAIL}")
    print(f"-> Admin Password: {RAW_ADMIN_PASSWORD}")
    print("=" * 70 + "\n")
    app.run(host=host, port=port, debug=debug)
