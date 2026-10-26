import json
import os
import sys
from typing import Dict, Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define project directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")#stores the trained model
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")#stores graphs and evaluation results.
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)#creates the folders if they don't exist.

# Set high-quality plotting aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


# ==============================================================================
# 1. DATA PROFILING AND INITIAL EXPLORATION
# ==============================================================================

def profile_raw_data(df: pd.DataFrame) -> None:
    """Displays dataset shape, columns, types, summary stats, nulls, duplicates, and inconsistent values."""
    print("\n" + "=" * 70)
    print("STEP 1: INITIAL DATA PROFILING (RAW DATASET)")
    print("=" * 70)

    print(f"\n1. First 5 Rows:\n{df.head().to_string()}")
    print(f"\n2. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\n3. Column Names: {list(df.columns)}")
    print(f"\n4. Data Types:\n{df.dtypes}")

    print("\n5. Summary Statistics (All Columns):")
    print(df.describe(include="all").to_string())

    print("\n6. Missing Values Check:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    print(f"\n7. Duplicate Records Count: {df.duplicated().sum()}")

    print("\n8. Inconsistent Category Value Counts:")
    print("Raw 'source' unique values:")
    print(df["source"].value_counts(dropna=False))
    print("\nRaw 'service_type' unique values:")
    print(df["service_type"].value_counts(dropna=False))

    # Check for invalid numeric values
    neg_resp = (df["response_time_hours"] < 0).sum()
    neg_follow = (df["previous_interactions_follow_up_count"] < 0).sum()
    print(f"\n9. Invalid Numeric Records: Negative response_time_hours={neg_resp}, Negative follow_ups={neg_follow}")


# ==============================================================================
# 2. DATA CLEANING AND IMPUTATION
# ==============================================================================

def normalize_category_value(val: Any) -> Any:
    """Normalizes categorical strings by stripping whitespace and standardizing casing."""
    if pd.isna(val):
        return np.nan
    val_clean = str(val).strip()
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
    return mapping.get(val_clean.lower(), val_clean.title())


def clean_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw dataframe:
    - Removes duplicates if present
    - Normalizes categorical labels
    - Replaces invalid negative numeric values with NaN
    - Creates a fully imputed cleaned working copy for EDA and records
    Returns (cleaned_working_df, normalized_df_for_modeling)
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 70)

    # Work on a separate copy to preserve raw data integrity
    clean_df = raw_df.copy()

    # 1. Deduplication
    dups_count = clean_df.duplicated().sum()
    if dups_count > 0:
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)
        print(f"-> Removed {dups_count} duplicate row(s).")
    else:
        print("-> Checked duplicates: None found.")

    # 2. Normalize categorical columns
    clean_df["source"] = clean_df["source"].apply(normalize_category_value)
    clean_df["service_type"] = clean_df["service_type"].apply(normalize_category_value)
    print("-> Normalized categorical labels (trimmed whitespace & unified title casing).")

    # 3. Handle invalid numeric values (set negative values to NaN for imputation)
    neg_resp_mask = clean_df["response_time_hours"] < 0
    if neg_resp_mask.sum() > 0:
        print(f"-> Detected {neg_resp_mask.sum()} negative response_time_hours entries; replacing with NaN.")
        clean_df.loc[neg_resp_mask, "response_time_hours"] = np.nan

    neg_follow_mask = clean_df["previous_interactions_follow_up_count"] < 0
    if neg_follow_mask.sum() > 0:
        print(f"-> Detected {neg_follow_mask.sum()} negative follow-up count entries; replacing with NaN.")
        clean_df.loc[neg_follow_mask, "previous_interactions_follow_up_count"] = np.nan

    # Create the standalone exported cleaned dataset with explicit imputation
    exported_clean_df = clean_df.copy()

    # Impute categorical with mode
    source_mode = exported_clean_df["source"].mode()[0]
    service_mode = exported_clean_df["service_type"].mode()[0]
    exported_clean_df["source"] = exported_clean_df["source"].fillna(source_mode)
    exported_clean_df["service_type"] = exported_clean_df["service_type"].fillna(service_mode)

    # Impute numeric with median
    budget_median = exported_clean_df["budget"].median()
    resp_median = exported_clean_df["response_time_hours"].median()
    follow_median = exported_clean_df["previous_interactions_follow_up_count"].median()

    exported_clean_df["budget"] = exported_clean_df["budget"].fillna(budget_median)
    exported_clean_df["response_time_hours"] = exported_clean_df["response_time_hours"].fillna(resp_median)
    exported_clean_df["previous_interactions_follow_up_count"] = (
        exported_clean_df["previous_interactions_follow_up_count"].fillna(follow_median).astype(int)
    )

    cleaned_csv_path = os.path.join(PROJECT_DIR, "customer_enquiries_cleaned.csv")
    exported_clean_df.to_csv(cleaned_csv_path, index=False)
    print(f"-> Saved cleaned dataset to: {cleaned_csv_path}")

    return exported_clean_df, clean_df


# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ==============================================================================

def generate_eda_visualizations(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """Generates all 6 required exploratory charts and saves them to outputs/."""
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 70)

    # Colors
    c_primary = "#2563EB"
    c_success = "#10B981"
    c_danger = "#EF4444"
    c_accent = "#F59E0B"
    palette = [c_danger, c_success]

    # Overall conversion rate
    total_enquiries = len(clean_df)
    conversions = clean_df["converted"].sum()
    conversion_rate = (conversions / total_enquiries) * 100
    print(f"Overall Dataset Conversion Rate: {conversion_rate:.2f}% ({conversions}/{total_enquiries} converted)")

    # -------------------------------------------------------------
    # Chart 1: Converted vs Not-Converted Enquiry Counts
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = clean_df["converted"].value_counts().sort_index()
    labels = ["Not Converted (0)", "Converted (1)"]
    bars = ax.bar(labels, counts.values, color=[c_danger, c_success], width=0.5, edgecolor="#1E293B", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_enquiries) * 100
        ax.annotate(
            f"{int(h)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_title("Overall Enquiry Conversion Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Enquiries", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    chart1_path = os.path.join(OUTPUTS_DIR, "eda_conversion_counts.png")
    fig.savefig(chart1_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {chart1_path}")

    # -------------------------------------------------------------
    # Chart 2: Conversion Rate by Enquiry Source
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    source_stats = (
        clean_df.groupby("source")["converted"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    bars = ax.bar(
        source_stats["source"],
        source_stats["mean"] * 100,
        color="#3B82F6",
        edgecolor="#1D4ED8",
        width=0.55,
    )
    for i, bar in enumerate(bars):
        h = bar.get_height()
        n = source_stats.iloc[i]["count"]
        ax.annotate(
            f"{h:.1f}%\n(n={int(n)})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Conversion Rate by Enquiry Source", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Conversion Rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    chart2_path = os.path.join(OUTPUTS_DIR, "eda_conversion_by_source.png")
    fig.savefig(chart2_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {chart2_path}")

    # -------------------------------------------------------------
    # Chart 3: Conversion Rate by Service Type
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    service_order = ["Basic", "Standard", "Premium", "Enterprise"]
    # Reindex to logical tier order
    service_stats = (
        clean_df.groupby("service_type")["converted"]
        .agg(["mean", "count"])
        .reindex([s for s in service_order if s in clean_df["service_type"].values])
        .reset_index()
    )
    bars = ax.bar(
        service_stats["service_type"],
        service_stats["mean"] * 100,
        color="#8B5CF6",
        edgecolor="#6D28D9",
        width=0.55,
    )
    for i, bar in enumerate(bars):
        h = bar.get_height()
        n = service_stats.iloc[i]["count"]
        ax.annotate(
            f"{h:.1f}%\n(n={int(n)})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Conversion Rate by Service Type", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Conversion Rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    chart3_path = os.path.join(OUTPUTS_DIR, "eda_conversion_by_service_type.png")
    fig.savefig(chart3_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {chart3_path}")

    # -------------------------------------------------------------
    # Chart 4: Budget Distribution by Conversion Status
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=clean_df,
        x="budget",
        hue="converted",
        palette={0: c_danger, 1: c_success},
        kde=True,
        bins=15,
        ax=ax,
        edgecolor="white",
        alpha=0.6,
    )
    ax.set_title("Customer Budget Distribution (Converted vs Not Converted)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Budget ($)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    # Update legend labels
    legend = ax.get_legend()
    if legend:
        legend.set_title("Conversion Status")
        for t, l in zip(legend.texts, ["Not Converted", "Converted"]):
            t.set_text(l)
    plt.tight_layout()
    chart4_path = os.path.join(OUTPUTS_DIR, "eda_budget_distribution.png")
    fig.savefig(chart4_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {chart4_path}")

    # -------------------------------------------------------------
    # Chart 5: Response Time vs Conversion Status
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    clean_df_resp = clean_df.copy()
    clean_df_resp["Conversion Status"] = clean_df_resp["converted"].map({0: "Not Converted", 1: "Converted"})
    sns.boxplot(
        data=clean_df_resp,
        x="Conversion Status",
        y="response_time_hours",
        hue="Conversion Status",
        palette={"Not Converted": c_danger, "Converted": c_success},
        legend=False,
        width=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=clean_df_resp,
        x="Conversion Status",
        y="response_time_hours",
        color="#0F172A",
        alpha=0.4,
        jitter=0.2,
        size=5,
        ax=ax,
    )
    ax.set_title("Response Time (Hours) vs Conversion Outcome", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Response Time (Hours)", fontsize=11)
    ax.set_xlabel("")
    plt.tight_layout()
    chart5_path = os.path.join(OUTPUTS_DIR, "eda_response_time_vs_conversion.png")
    fig.savefig(chart5_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {chart5_path}")

    # -------------------------------------------------------------
    # Chart 6: Missing Values Before vs After Cleaning
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    columns_to_show = ["source", "service_type", "budget", "response_time_hours", "previous_interactions_follow_up_count"]
    before_missing = [raw_df[col].isnull().sum() for col in columns_to_show]
    after_missing = [clean_df[col].isnull().sum() for col in columns_to_show]

    x = np.arange(len(columns_to_show))
    width = 0.35

    rects1 = ax.bar(x - width / 2, before_missing, width, label="Before Cleaning (Raw)", color="#F97316")
    rects2 = ax.bar(x + width / 2, after_missing, width, label="After Cleaning (Imputed)", color="#10B981")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Missing Value Count: Before vs After Data Cleaning", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Source", "Service Type", "Budget", "Response Time", "Follow-ups"],
        rotation=15,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Missing Record Count", fontsize=11)
    ax.set_ylim(0, max(before_missing + [1]) * 1.35)
    ax.legend(frameon=True)
    plt.tight_layout()
    chart6_path = os.path.join(OUTPUTS_DIR, "eda_missing_values_before_after.png")
    fig.savefig(chart6_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {chart6_path}")


# ==============================================================================
# 4. MODEL DEVELOPMENT, TRAINING & EVALUATION
# ==============================================================================

def train_and_evaluate_models(clean_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Builds Scikit-learn preprocessing ColumnTransformer and classification Pipelines.
    Trains Logistic Regression and Random Forest.
    Evaluates both models on held-out test set and saves best model.
    """
    print("\n" + "=" * 70)
    print("STEP 4: MODEL DEVELOPMENT AND EVALUATION")
    print("=" * 70)

    # 1. Feature selection: Drop enquiry_id (unique identifier)
    feature_cols = [
        "source",
        "service_type",
        "budget",
        "response_time_hours",
        "previous_interactions_follow_up_count",
    ]
    target_col = "converted"

    X = clean_df[feature_cols].copy()
    y = clean_df[target_col].copy()

    # 2. Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Total dataset: {len(X)} records")
    print(f"Training split: {len(X_train)} records (Target: {y_train.sum()} converted, {len(y_train)-y_train.sum()} not converted)")
    print(f"Test split:     {len(X_test)} records (Target: {y_test.sum()} converted, {len(y_test)-y_test.sum()} not converted)")

    # 3. ColumnTransformer for Preprocessing
    categorical_features = ["source", "service_type"]
    numerical_features = ["budget", "response_time_hours", "previous_interactions_follow_up_count"]

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    # 4. Candidate Models
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000, C=1.0)),
            ]
        ),
        "Random Forest Classifier": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=4,
                        min_samples_split=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results = {}

    for name, pipe in models.items():
        print(f"\n--- Training {name} ---")
        # Fit pipeline strictly on training data (no data leakage)
        pipe.fit(X_train, y_train)

        # Predictions on test data
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            "pipeline": pipe,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # -------------------------------------------------------------
    # 5. Model Comparison Visualizations
    # -------------------------------------------------------------
    # Bar chart comparing metrics
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics_names))
    width = 0.35

    lr_vals = [
        results["Logistic Regression"]["accuracy"],
        results["Logistic Regression"]["precision"],
        results["Logistic Regression"]["recall"],
        results["Logistic Regression"]["f1_score"],
        results["Logistic Regression"]["roc_auc"],
    ]
    rf_vals = [
        results["Random Forest Classifier"]["accuracy"],
        results["Random Forest Classifier"]["precision"],
        results["Random Forest Classifier"]["recall"],
        results["Random Forest Classifier"]["f1_score"],
        results["Random Forest Classifier"]["roc_auc"],
    ]

    r1 = ax.bar(x - width / 2, lr_vals, width, label="Logistic Regression", color="#3B82F6", edgecolor="#1D4ED8")
    r2 = ax.bar(x + width / 2, rf_vals, width, label="Random Forest", color="#10B981", edgecolor="#047857")

    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Model Comparison on Held-Out Test Set (20% Split)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    comparison_chart_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    fig.savefig(comparison_chart_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {comparison_chart_path}")

    # Confusion Matrices Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (name, res) in enumerate(results.items()):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if i == 0 else "Greens",
            cbar=False,
            ax=axes[i],
            xticklabels=["Not Converted (0)", "Converted (1)"],
            yticklabels=["Not Converted (0)", "Converted (1)"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        axes[i].set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Predicted Class", fontsize=10)
        axes[i].set_ylabel("Actual Class", fontsize=10)

    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "model_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {cm_path}")

    # ROC Curves Plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, res, col in zip(
        ["Logistic Regression", "Random Forest Classifier"],
        [results["Logistic Regression"], results["Random Forest Classifier"]],
        ["#2563EB", "#10B981"],
    ):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.2f})", color=col, linewidth=2.2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
    ax.set_title("ROC Curves Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUTS_DIR, "model_roc_curve.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved: {roc_path}")

    # -------------------------------------------------------------
    # 6. Model Selection and Artifact Export
    # -------------------------------------------------------------
    # Select best model based on primary metric: F1-score (or ROC-AUC)
    # Both metrics favor balanced performance on conversion class
    best_model_name = max(results.keys(), key=lambda k: (results[k]["f1_score"], results[k]["roc_auc"]))
    best_pipeline = results[best_model_name]["pipeline"]

    print(f"\n========================================================")
    print(f"SELECTED BEST MODEL: {best_model_name}")
    print(f"Selected based on balanced F1-score ({results[best_model_name]['f1_score']:.4f}) and ROC-AUC ({results[best_model_name]['roc_auc']:.4f})")
    print("========================================================")

    # Save best model pipeline
    model_save_path = os.path.join(MODELS_DIR, "customer_conversion_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"-> Saved model pipeline to: {model_save_path}")

    # Save metrics JSON
    metrics_summary = {
        "best_model": best_model_name,
        "sample_size": len(clean_df),
        "test_size": len(X_test),
        "note": "Metrics on this 100-sample synthetic dataset are illustrative for demonstration.",
        "models": {
            name: {
                "accuracy": res["accuracy"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1_score": res["f1_score"],
                "roc_auc": res["roc_auc"],
                "confusion_matrix": res["confusion_matrix"],
            }
            for name, res in results.items()
        },
    }

    metrics_json_path = os.path.join(OUTPUTS_DIR, "model_evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"-> Saved metrics summary to: {metrics_json_path}")

    return best_pipeline, metrics_summary


# ==============================================================================
# 5. PREDICTION FUNCTION FOR NEW CUSTOMERS
# ==============================================================================

def predict_conversion(
    model_pipeline: Any,
    enquiry_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Accepts a single new enquiry as a Pandas DataFrame with exact feature column names:
    - 'source'
    - 'service_type'
    - 'budget'
    - 'response_time_hours'
    - 'previous_interactions_follow_up_count'

    Returns a structured dictionary with:
    - predicted_class: 'Converted' or 'Not Converted'
    - predicted_label: 1 or 0
    - conversion_probability_pct: Model estimated probability percentage
    - disclaimer: Clear indicator that probability is an estimate, not a guarantee.
    """
    expected_cols = [
        "source",
        "service_type",
        "budget",
        "response_time_hours",
        "previous_interactions_follow_up_count",
    ]

    # Validate DataFrame schema
    missing_cols = [c for c in expected_cols if c not in enquiry_df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing_cols}")

    # Normalize category entries in input
    clean_input = enquiry_df[expected_cols].copy()
    clean_input["source"] = clean_input["source"].apply(normalize_category_value)
    clean_input["service_type"] = clean_input["service_type"].apply(normalize_category_value)

    # Predict label and probability
    pred_label = int(model_pipeline.predict(clean_input)[0])
    probabilities = model_pipeline.predict_proba(clean_input)[0]
    conversion_prob = float(probabilities[1]) * 100.0

    pred_class = "Converted" if pred_label == 1 else "Not Converted"

    return {
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "conversion_probability_pct": round(conversion_prob, 2),
        "disclaimer": (
            "Model estimate based on historical patterns. "
            "This probability is an indicative estimate for follow-up prioritization, not a guaranteed outcome."
        ),
    }


# ==============================================================================
# MAIN WORKFLOW EXECUTION
# ==============================================================================

def main():
    raw_data_path = os.path.join(PROJECT_DIR, "customer_enquiries.csv")
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found. Running generate_data.py first...")
        from generate_data import generate_synthetic_dataset
        raw_df = generate_synthetic_dataset(100, 42)
        raw_df.to_csv(raw_data_path, index=False)
    else:
        raw_df = pd.read_csv(raw_data_path)

    # Step 1: Profiling
    profile_raw_data(raw_df)

    # Step 2: Cleaning & Imputation
    clean_exported_df, clean_modeling_df = clean_dataset(raw_df)

    # Step 3: EDA
    generate_eda_visualizations(raw_df, clean_exported_df)

    # Step 4: Model Development & Evaluation
    best_pipe, _ = train_and_evaluate_models(clean_exported_df)

    # Step 5: Test Prediction Function on Sample Prospects
    print("\n" + "=" * 70)
    print("STEP 5: DEMONSTRATING NEW CUSTOMER PREDICTION FUNCTION")
    print("=" * 70)

    # Sample 1: Strong Referral prospect
    sample_lead_1 = pd.DataFrame([{
        "source": "Referral",
        "service_type": "Enterprise",
        "budget": 25000.0,
        "response_time_hours": 2.0,
        "previous_interactions_follow_up_count": 4,
    }])
    result_1 = predict_conversion(best_pipe, sample_lead_1)
    print("\n--- Test Sample 1 (High Potential Referral) ---")
    print(f"Inputs:\n{sample_lead_1.to_dict(orient='records')[0]}")
    print(f"Predicted Class: {result_1['predicted_class']} (Label: {result_1['predicted_label']})")
    print(f"Conversion Probability: {result_1['conversion_probability_pct']}%")
    print(f"Note: {result_1['disclaimer']}")

    # Sample 2: Cold Social Media lead with slow response
    sample_lead_2 = pd.DataFrame([{
        "source": "social media ",
        "service_type": "basic",
        "budget": 950.0,
        "response_time_hours": 36.0,
        "previous_interactions_follow_up_count": 0,
    }])
    result_2 = predict_conversion(best_pipe, sample_lead_2)
    print("\n--- Test Sample 2 (Cold Lead with Inconsistent Casing & Slow Response) ---")
    print(f"Inputs:\n{sample_lead_2.to_dict(orient='records')[0]}")
    print(f"Predicted Class: {result_2['predicted_class']} (Label: {result_2['predicted_label']})")
    print(f"Conversion Probability: {result_2['conversion_probability_pct']}%")
    print(f"Note: {result_2['disclaimer']}")

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
