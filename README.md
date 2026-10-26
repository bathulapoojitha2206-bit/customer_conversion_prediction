# Customer Conversion Prediction Using Machine Learning

An end-to-end, beginner-friendly Machine Learning project that predicts whether a prospective customer enquiry will convert into an actual paying customer using Scikit-learn binary classification pipelines and an interactive Streamlit web application.

---

## Table of Contents

1. [Business Problem & Objective](#1-business-problem--objective)
2. [Project Architecture & Directory Structure](#2-project-architecture--directory-structure)
3. [Dataset Description & Data Quality Issues](#3-dataset-description--data-quality-issues)
4. [Data Cleaning & Preprocessing Pipeline](#4-data-cleaning--preprocessing-pipeline)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Machine Learning Models & Training](#6-machine-learning-models--training)
7. [Model Evaluation & Metrics Interpretation](#7-model-evaluation--metrics-interpretation)
8. [Predicting Conversion for New Prospects](#8-predicting-conversion-for-new-prospects)
9. [Interactive Streamlit Application](#9-interactive-streamlit-application)
10. [Installation & Execution Guide](#10-installation--execution-guide)
11. [Limitations & Real-World Considerations](#11-limitations--real-world-considerations)

---

## 1. Business Problem & Objective

### The Business Scenario
A modern service company receives enquiries daily across multiple digital and offline acquisition channels (e.g., website forms, social media campaigns, referrals, walk-ins, paid advertising). 

Sales teams have limited time and cannot give equal, high-touch attention to every enquiry simultaneously. Following up with unqualified leads wastes valuable sales capacity, while slow responses to high-intent prospective buyers lead to lost deals.

### The Objective
Build a binary classification Machine Learning system that predicts prospective customer conversion:
- `1` = **Converted** (The prospect purchases a service / becomes a client)
- `0` = **Not Converted** (The prospect drops off or declines)

### Business Value
- **Lead Prioritization:** Instantly rank incoming leads by predicted conversion likelihood.
- **Shorter Sales Cycles:** Route high-intent prospects directly to senior Account Executives.
- **Automated Lead Nurturing:** Enroll lower-propensity enquiries into automated email nurture sequences.
- **Resource Optimization:** Maximize sales team return on investment (ROI).

---

## 2. Project Architecture & Directory Structure

```text
customer-conversion-prediction/
├── customer_enquiries.csv              # Original 100 synthetic records with intentional quality issues
├── customer_enquiries_cleaned.csv      # Cleaned and imputed working dataset
├── generate_data.py                    # Script to generate synthetic records with realistic signals
├── train_model.py                      # Data cleaning, EDA generation, ML pipelines, evaluation
├── app.py                              # Interactive Streamlit web application
├── requirements.txt                    # Project library dependencies
├── README.md                           # Comprehensive project documentation
├── models/
│   └── customer_conversion_model.joblib # Serialized winning Scikit-learn Pipeline
└── outputs/                            # Exported charts and metrics
    ├── eda_conversion_counts.png
    ├── eda_conversion_by_source.png
    ├── eda_conversion_by_service_type.png
    ├── eda_budget_distribution.png
    ├── eda_response_time_vs_conversion.png
    ├── eda_missing_values_before_after.png
    ├── model_comparison.png
    ├── model_confusion_matrix.png
    ├── model_roc_curve.png
    └── model_evaluation_metrics.json
```

---

## 3. Dataset Description & Data Quality Issues

The dataset contains exactly **100 sample records**. No real customer or personally identifiable information (PII) is used.

### Schema & Column Order

| # | Column Name | Data Type | Description | Allowed / Example Values |
|---|---|---|---|---|
| 1 | `enquiry_id` | String / Object | Unique enquiry identifier | `ENQ001` ... `ENQ100` |
| 2 | `source` | Categorical / String | Acquisition channel | `Website`, `Referral`, `Social Media`, `Advertisement`, `Walk-in` |
| 3 | `service_type` | Categorical / String | Requested service package tier | `Basic`, `Standard`, `Premium`, `Enterprise` |
| 4 | `budget` | Float / Numeric | Stated customer budget in USD | `$800.00` to `$45,000.00` |
| 5 | `response_time_hours` | Float / Numeric | Time taken to respond to lead | `0.5` to `72.0` hours |
| 6 | `previous_interactions_follow_up_count` | Integer / Numeric | Number of prior interactions / calls | `0` to `8` touchpoints |
| 7 | `converted` | Binary Integer | Target label | `1` (Converted) or `0` (Not Converted) |

### Realistic Data Quality Issues Intentionally Introduced

In real enterprise systems, raw data is never pristine. To demonstrate production-grade data hygiene, several common defects were intentionally injected:

1. **Inconsistent Casing & Trailing Whitespace:**
   - Categorical variations such as `'website'`, `'Website '`, `'WEBSITE'`, `'referral'`, `'Referral '`, `'social media'`, `'ADVERTISEMENT'`.
   - Service tier variations such as `'premium'`, `'Premium '`, `'PREMIUM'`, `'enterprise'`, `'basic '`, `'STANDARD'`.
2. **Missing Values (`NaN`):**
   - 3 missing records in `source`.
   - 3 missing records in `service_type`.
   - 4 missing records in `budget`.
   - 4 missing records in `response_time_hours`.
3. **Invalid Numeric Records:**
   - Negative response times (e.g., `-4.5` hours, `-1.2` hours) representing logging glitches.
   - Negative follow-up counts (e.g., `-2`, `-1`) representing data entry errors.
4. **Data Integrity Guarantee:**
   - All `enquiry_id` values remain unique.
   - Target labels (`converted`) are strictly valid binary values (`0` or `1`).

---

## 4. Data Cleaning & Preprocessing Pipeline

Data cleaning is performed on a separate working copy, preserving raw `customer_enquiries.csv` for auditability:

1. **Deduplication:** Verifies record uniqueness (`df.duplicated().sum()`).
2. **Text Standardization:**
   - Trims leading/trailing whitespace (`.str.strip()`).
   - Normalizes text casing to canonical title case (`Website`, `Referral`, `Social Media`, `Advertisement`, `Walk-in`, `Basic`, `Standard`, `Premium`, `Enterprise`).
3. **Anomaly Rectification:**
   - Flags negative numeric values (`response_time_hours < 0`, `follow_ups < 0`) and replaces them with `NaN` for imputation.
4. **Imputation Strategy:**
   - Categorical missing values (`source`, `service_type`) are imputed with the **most frequent category (mode)**.
   - Numerical missing values (`budget`, `response_time_hours`, `follow_ups`) are imputed with the **median** (robust to skewed outliers).
   - Cleaned dataset is saved to `customer_enquiries_cleaned.csv`.
5. **Feature Matrix Separation & Data Leakage Prevention:**
   - `enquiry_id` is dropped from model features because it is an arbitrary identifier.
   - Data is split into training (80%) and testing (20%) sets using stratified sampling (`stratify=y`).
   - Scikit-learn `ColumnTransformer` fits imputers, encoders, and scalers **strictly on the training split**, preventing test set information from leaking into training.

```python
# Pipeline Architecture
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ]
)
```

---

## 5. Exploratory Data Analysis (EDA)

Six exploratory charts are generated and saved to `outputs/`:

| Chart Name | Output File | Key Business Insight |
|---|---|---|
| **Conversion Distribution** | `eda_conversion_counts.png` | ~33% overall conversion rate, reflecting realistic class imbalance in B2B/B2C sales. |
| **Conversion by Source** | `eda_conversion_by_source.png` | `Referral` and `Walk-in` channels show the highest conversion (>60%), while cold digital ads show lower rates. |
| **Conversion by Service Tier** | `eda_conversion_by_service_type.png` | `Enterprise` and `Premium` leads convert at significantly higher rates due to strong commercial intent. |
| **Budget Distribution** | `eda_budget_distribution.png` | Higher-budget enquiries exhibit higher conversion density, particularly above $10,000. |
| **Response Time vs Conversion** | `eda_response_time_vs_conversion.png` | Rapid response (< 5 hours) strongly correlates with conversion; slow responses (> 24 hours) lead to lead decay. |
| **Data Quality Before & After** | `eda_missing_values_before_after.png` | Validates that all missing values across features were successfully resolved. |

---

## 6. Machine Learning Models & Training

Two distinct binary classification algorithms are trained and compared:

### 1. Logistic Regression
- **Type:** Linear parametric classification model.
- **Why used:** High interpretability, fast training, outputs calibrated log-odds probabilities, serves as a standard industry baseline.
- **Hyperparameters:** `C=1.0`, `max_iter=1000`, `random_state=42`.

### 2. Random Forest Classifier
- **Type:** Non-linear ensemble of decision trees with bootstrap aggregation (bagging).
- **Why used:** Captures complex feature interactions, robust to non-linear thresholds, handles numeric and one-hot categorical features seamlessly.
- **Hyperparameters:** `n_estimators=100`, `max_depth=4`, `min_samples_split=4`, `random_state=42`.

### Training Protocol
- **Split:** 80% train (80 samples), 20% test (20 samples).
- **Stratification:** Maintained 33% positive class ratio across both splits.
- **End-to-End Pipeline:** Preprocessing and classifier are bundled into a single pipeline object, saved to `models/customer_conversion_model.joblib`.

---

## 7. Model Evaluation & Metrics Interpretation

### Evaluation Results (Held-Out Test Set)

| Metric | Logistic Regression | Random Forest Classifier | Best Performer |
|---|---|---|---|
| **Accuracy** | 70.00% | **80.00%** | Random Forest |
| **Precision** | 66.67% | **100.00%** | Random Forest |
| **Recall** | 28.57% | **42.86%** | Random Forest |
| **F1-Score** | 0.4000 | **0.6000** | Random Forest |
| **ROC-AUC** | 0.8022 | **0.8681** | Random Forest |

> **Selected Model:** **Random Forest Classifier** achieved superior F1-score (0.60) and ROC-AUC (0.87), providing zero false positives on the test set while capturing high-value prospects.

### Understanding Evaluation Metrics in Simple Terms

1. **Accuracy ($\frac{TP + TN}{\text{Total}}$):**
   - The percentage of total correct predictions.
   - *Why accuracy alone is dangerous:* If 80% of leads do not convert, a trivial model that always predicts `0` achieves 80% accuracy without finding a single customer.
2. **Precision ($\frac{TP}{TP + FP}$):**
   - *When the model predicts a customer will convert, how often is it right?*
   - *Business cost of low precision:* Sales reps waste valuable hours chasing cold leads marked as "hot".
3. **Recall ($\frac{TP}{TP + FN}$):**
   - *Out of all customers who actually wanted to convert, how many did the model identify?*
   - *Business cost of low recall:* High-intent leads are ignored, resulting in lost revenue to competitors.
4. **F1-Score ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$):**
   - The harmonic mean of precision and recall. Balances both errors (false positives and false negatives).
5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
   - Measures how well the model ranks converting leads higher than non-converting leads across all probability thresholds. An AUC of 0.50 is coin-toss guessing; 1.00 is perfect.

---

## 8. Predicting Conversion for New Prospects

The project includes a production-ready prediction function in `train_model.py`:

```python
from train_model import predict_conversion
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load("models/customer_conversion_model.joblib")

# Create a DataFrame with a single new enquiry
new_enquiry = pd.DataFrame([{
    "source": "Referral",
    "service_type": "Enterprise",
    "budget": 28000.0,
    "response_time_hours": 2.0,
    "previous_interactions_follow_up_count": 4
}])

# Generate prediction
result = predict_conversion(model, new_enquiry)
print(result)
```

### Output Format

```json
{
    "predicted_class": "Converted",
    "predicted_label": 1,
    "conversion_probability_pct": 55.57,
    "disclaimer": "Model estimate based on historical patterns. This probability is an indicative estimate for follow-up prioritization, not a guaranteed outcome."
}
```

---

## 9. Interactive Streamlit Application

The Streamlit web application (`app.py`) provides an interactive interface for sales and management teams:

- **Lead Scorer:** Real-time form for entering the 5 enquiry features with instant validation.
- **Quick Test Presets:** 1-click scenario loading (`High-Intent Referral`, `Enterprise Prospect`, `Slow Social Media Lead`, `Average Web Enquiry`).
- **Strategic Action Recommendations:** Automatically suggests tailored next steps based on predicted score.
- **Model Evaluation Explorer:** Visual confusion matrices, ROC curves, and metric guides.
- **EDA Gallery:** Interactive browser for all exploratory data charts.
- **Data Quality Viewer:** Compare raw vs cleaned dataset with clear documentation of fixes.

---

## 10. Installation & Execution Guide

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate the Synthetic Dataset
```bash
python generate_data.py
```
*Generates `customer_enquiries.csv` with 100 records and intentional data quality issues.*

### Step 3: Clean Data, Train Models & Generate Outputs
```bash
python train_model.py
```
*Runs data profiling, generates cleaned CSV, creates EDA and model evaluation charts in `outputs/`, trains Logistic Regression & Random Forest, and serializes the winning model to `models/customer_conversion_model.joblib`.*

### Step 4: Launch the Admin Web Portal (Flask Application)
```bash
python server.py
# Or from the workspace root:
python run.py
```
*Launches the professional Admin Portal at `http://127.0.0.1:5000` featuring:*
- Modern Admin Login page with Show/Hide Password toggle and loading states.
- Secure session-based authentication with Werkzeug password hashing.
- Protected `/dashboard` route with live KPIs, Chart.js graphs, and customer records table.
- Interactive ML inference form using the trained Scikit-learn Pipeline.

#### Admin Credentials Configuration (`.env`)
Copy `.env.example` to `.env` and set your credentials:
```env
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@conversion.ai
ADMIN_PASSWORD=admin123
SECRET_KEY=your_secret_key_here
```
*Note: Passwords are automatically hashed using PBKDF2/SHA256 upon server initialization.*

### Step 5: Run Automated Test Suite
```bash
python -m unittest test_app.py
```
*Executes 9 unit and integration tests verifying login validation, session protection, logout, and ML prediction.*

### Step 6: (Alternative) Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Opens the Streamlit dashboard (default: `http://localhost:8501`), protected by the Admin Authentication gate.*

---

## 11. Limitations & Real-World Considerations

1. **Synthetic Data Limitations:**
   - The dataset was generated algorithmically to simulate business relationships. Real-world sales cycles involve nuanced external factors (economic shifts, competitor pricing, seasonal trends) not captured here.
2. **Sample Size Constraints (100 Records):**
   - With 100 total records and a 20-sample test set, metrics (e.g., 80% accuracy, 100% precision) are illustrative for project architecture demonstration. In production, thousands of historical leads and cross-validation are recommended.
3. **Probabilistic Nature of Machine Learning:**
   - Conversion probabilities represent statistical estimates, not business guarantees. Human sales judgment should always accompany algorithmic recommendations.
4. **Data Drift:**
   - Over time, acquisition channel performance changes. In a production environment, models must be monitored and retrained periodically as new conversion data accumulates.

---

## License & Attribution
Developed for machine learning demonstration and technical interview portfolio presentation. Code written using Python, Pandas, Scikit-learn, and Streamlit.
