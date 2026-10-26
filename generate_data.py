"""
generate_data.py
----------------
Generates a realistic synthetic dataset of prospective customer enquiries (100 records)
with intentional, realistic data quality issues for demonstrating data cleaning and
machine learning model development.

Columns (in exact specified order):
1. enquiry_id
2. source
3. service_type
4. budget
5. response_time_hours
6. previous_interactions_follow_up_count
7. converted
"""

import numpy as np
import pandas as pd


def generate_synthetic_dataset(num_records: int = 100, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer enquiry dataset with realistic business signals
    and intentional data quality issues.
    """
    np.random.seed(random_seed)

    # 1. Unique enquiry IDs
    enquiry_ids = [f"ENQ{i:03d}" for i in range(1, num_records + 1)]

    # 2. Categorical features
    sources = ["Website", "Referral", "Social Media", "Advertisement", "Walk-in"]
    source_weights = [0.35, 0.20, 0.20, 0.15, 0.10]
    sampled_sources = np.random.choice(sources, size=num_records, p=source_weights)

    service_types = ["Basic", "Standard", "Premium", "Enterprise"]
    service_weights = [0.30, 0.40, 0.20, 0.10]
    sampled_services = np.random.choice(service_types, size=num_records, p=service_weights)

    # 3. Numeric features aligned with service types
    budget_map = {
        "Basic": (800, 2500),
        "Standard": (2500, 7500),
        "Premium": (7500, 18000),
        "Enterprise": (18000, 45000),
    }

    budgets = []
    response_times = []
    follow_up_counts = []

    for i in range(num_records):
        st = sampled_services[i]
        b_low, b_high = budget_map[st]
        # Sample budget with variation
        budget = round(float(np.random.uniform(b_low, b_high)), 2)
        budgets.append(budget)

        # Response time in hours (typically 0.5 to 48 hours, log-normal distribution)
        resp_time = round(float(np.random.exponential(scale=12.0) + 0.5), 1)
        response_times.append(min(resp_time, 72.0))

        # Previous interactions / follow-up count (0 to 7)
        follow_ups = int(np.random.poisson(lam=2.5))
        follow_up_counts.append(min(follow_ups, 8))

    budgets = np.array(budgets)
    response_times = np.array(response_times)
    follow_up_counts = np.array(follow_up_counts)

    # 4. Realistic conversion probability calculation (ground truth signal + noise)
    source_score = {
        "Referral": 1.5,
        "Walk-in": 1.1,
        "Website": 0.2,
        "Advertisement": -0.3,
        "Social Media": -0.4,
    }

    service_score = {
        "Enterprise": 0.8,
        "Premium": 0.5,
        "Standard": 0.1,
        "Basic": -0.2,
    }

    # Log-odds formula
    z = (
        -1.2  # Base intercept
        + np.array([source_score[s] for s in sampled_sources])
        + np.array([service_score[st] for st in sampled_services])
        + 0.35 * (follow_up_counts - 2.0)  # Follow-ups help conversion
        - 0.05 * (response_times - 10.0)   # Faster response improves conversion
        + 0.00003 * (budgets - 10000.0)     # Slight positive budget effect
    )

    # Sigmoid function for probability
    probabilities = 1.0 / (1.0 + np.exp(-z))
    # Clip to avoid deterministic values
    probabilities = np.clip(probabilities, 0.05, 0.95)

    # Generate binary conversion target (0 or 1)
    converted = (np.random.uniform(0, 1, size=num_records) < probabilities).astype(int)

    # Create base clean dataframe
    df = pd.DataFrame({
        "enquiry_id": enquiry_ids,
        "source": sampled_sources,
        "service_type": sampled_services,
        "budget": budgets,
        "response_time_hours": response_times,
        "previous_interactions_follow_up_count": follow_up_counts,
        "converted": converted,
    })

    # -------------------------------------------------------------
    # 5. Intentionally introduce realistic data quality issues
    # -------------------------------------------------------------

    # A. Inconsistent categorical casing and trailing whitespace
    # For 'source':
    noisy_sources = {
        3: "website",
        11: "Website ",
        19: "WEBSITE",
        27: "referral",
        35: "Referral ",
        48: "social media",
        62: "ADVERTISEMENT",
        75: "walk-in ",
    }
    for idx, val in noisy_sources.items():
        if idx < num_records:
            df.at[idx, "source"] = val

    # For 'service_type':
    noisy_services = {
        5: "premium",
        14: "Premium ",
        22: "PREMIUM",
        31: "enterprise",
        42: "Enterprise ",
        55: "basic ",
        69: "STANDARD",
        84: "standard ",
    }
    for idx, val in noisy_services.items():
        if idx < num_records:
            df.at[idx, "service_type"] = val

    # B. Missing values (NaN) in source, service_type, budget, response_time_hours
    # Introduce ~3 to 4 missing values in each specified column
    missing_source_indices = [8, 44, 78]
    missing_service_indices = [16, 52, 88]
    missing_budget_indices = [12, 38, 67, 92]
    missing_response_time_indices = [6, 29, 61, 83]

    for idx in missing_source_indices:
        df.at[idx, "source"] = np.nan

    for idx in missing_service_indices:
        df.at[idx, "service_type"] = np.nan

    for idx in missing_budget_indices:
        df.at[idx, "budget"] = np.nan

    for idx in missing_response_time_indices:
        df.at[idx, "response_time_hours"] = np.nan

    # C. Invalid numeric values (negative response times and negative follow-up counts)
    df.at[24, "response_time_hours"] = -4.5
    df.at[71, "response_time_hours"] = -1.2
    df.at[17, "previous_interactions_follow_up_count"] = -2
    df.at[59, "previous_interactions_follow_up_count"] = -1

    # Ensure enquiry_id is unique and target column contains valid binary labels
    assert df["enquiry_id"].nunique() == num_records, "enquiry_id must be unique"
    assert set(df["converted"].unique()).issubset({0, 1}), "converted must be binary (0 or 1)"
    assert len(df) == num_records, f"Dataset must contain exactly {num_records} records"

    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(num_records=100, random_seed=42)
    output_path = "customer_enquiries.csv"
    df.to_csv(output_path, index=False)

    print("=" * 60)
    print("Synthetic Customer Enquiries Dataset Generated Successfully")
    print("=" * 60)
    print(f"File saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df['converted'].value_counts(normalize=True)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nSample records with intentional data quality issues:")
    print(df.iloc[[3, 6, 8, 11, 12, 17, 24]].to_string())
    print("=" * 60)
