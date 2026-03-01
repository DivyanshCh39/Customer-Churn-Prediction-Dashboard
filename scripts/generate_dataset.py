"""
Customer Churn Dataset Generator
Telecom/OTT Company - TeleVision Plus (Dummy Dataset)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

n_customers = 5000

# Customer IDs
customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

# Demographics
ages = np.random.normal(38, 12, n_customers).clip(18, 75).astype(int)
genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04])
cities = np.random.choice(
    ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'],
    n_customers, p=[0.18, 0.17, 0.15, 0.10, 0.10, 0.08, 0.08, 0.06, 0.04, 0.04]
)
states = {
    'Mumbai': 'Maharashtra', 'Delhi': 'Delhi', 'Bangalore': 'Karnataka',
    'Hyderabad': 'Telangana', 'Chennai': 'Tamil Nadu', 'Kolkata': 'West Bengal',
    'Pune': 'Maharashtra', 'Ahmedabad': 'Gujarat', 'Jaipur': 'Rajasthan', 'Lucknow': 'Uttar Pradesh'
}
state_col = [states[c] for c in cities]

# Subscription details
plans = np.random.choice(['Basic', 'Standard', 'Premium', 'Enterprise'], n_customers, p=[0.30, 0.35, 0.25, 0.10])
plan_price = {'Basic': 199, 'Standard': 399, 'Premium': 699, 'Enterprise': 1299}
monthly_charges = np.array([plan_price[p] for p in plans]) + np.random.normal(0, 30, n_customers)
monthly_charges = monthly_charges.clip(150, 1500).round(2)

tenure_months = np.random.exponential(24, n_customers).clip(1, 72).astype(int)
total_charges = (monthly_charges * tenure_months + np.random.normal(0, 100, n_customers)).clip(0).round(2)

contract_types = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers, p=[0.50, 0.30, 0.20])
payment_methods = np.random.choice(
    ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Auto-Debit'],
    n_customers, p=[0.25, 0.20, 0.30, 0.15, 0.10]
)

# Services
internet_service = np.random.choice(['Fiber Optic', 'Cable', 'DSL', 'No'], n_customers, p=[0.40, 0.25, 0.25, 0.10])
streaming_tv = np.random.choice(['Yes', 'No'], n_customers, p=[0.60, 0.40])
streaming_movies = np.random.choice(['Yes', 'No'], n_customers, p=[0.55, 0.45])
online_backup = np.random.choice(['Yes', 'No'], n_customers, p=[0.45, 0.55])
tech_support = np.random.choice(['Yes', 'No'], n_customers, p=[0.40, 0.60])
multiple_lines = np.random.choice(['Yes', 'No', 'No Phone Service'], n_customers, p=[0.42, 0.48, 0.10])
device_protection = np.random.choice(['Yes', 'No'], n_customers, p=[0.35, 0.65])
paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.60, 0.40])

# Usage metrics
avg_monthly_data_gb = np.random.lognormal(3.5, 0.8, n_customers).clip(1, 500).round(1)
avg_call_minutes = np.random.normal(300, 150, n_customers).clip(0, 1200).round(0)
support_calls_last_6m = np.random.poisson(1.5, n_customers).clip(0, 10)
complaints_last_year = np.random.poisson(0.8, n_customers).clip(0, 8)
network_downtime_hours = np.random.exponential(2, n_customers).clip(0, 20).round(1)
avg_rating = np.random.normal(3.5, 1.0, n_customers).clip(1, 5).round(1)
payment_delays = np.random.poisson(0.5, n_customers).clip(0, 6)

# Dependents and partner
has_partner = np.random.choice(['Yes', 'No'], n_customers, p=[0.48, 0.52])
has_dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70])
is_senior_citizen = (ages >= 60).astype(int)

# Churn probability calculation (realistic logic)
churn_score = np.zeros(n_customers)

# Higher churn for month-to-month
churn_score += np.where(np.array(contract_types) == 'Month-to-Month', 0.25, 0)
churn_score += np.where(np.array(contract_types) == 'One Year', 0.10, 0)

# Lower tenure = higher churn
churn_score += np.where(tenure_months < 6, 0.20, 0)
churn_score += np.where((tenure_months >= 6) & (tenure_months < 12), 0.10, 0)
churn_score += np.where(tenure_months > 36, -0.15, 0)

# More complaints = higher churn
churn_score += complaints_last_year * 0.08
churn_score += support_calls_last_6m * 0.04

# Low rating = higher churn
churn_score += np.where(avg_rating < 2.5, 0.20, 0)
churn_score += np.where(avg_rating > 4.0, -0.10, 0)

# Payment delays
churn_score += payment_delays * 0.05

# High charges
churn_score += np.where(monthly_charges > 800, 0.10, 0)

# Network issues
churn_score += np.where(network_downtime_hours > 10, 0.15, 0)

# No tech support + fiber = slightly higher churn
churn_score += np.where((np.array(tech_support) == 'No') & (np.array(internet_service) == 'Fiber Optic'), 0.08, 0)

# Senior citizen
churn_score += np.where(is_senior_citizen == 1, 0.05, 0)

# Normalize to probability
churn_prob = 1 / (1 + np.exp(-3 * (churn_score - 0.5)))
churn_prob = churn_prob.clip(0.02, 0.95)

# Generate churn labels
churn = (np.random.random(n_customers) < churn_prob).astype(int)

# Join dates
base_date = datetime(2025, 1, 1)
join_dates = [base_date - timedelta(days=int(t * 30)) for t in tenure_months]
join_date_str = [d.strftime('%Y-%m-%d') for d in join_dates]

# Last interaction date
last_interaction = [
    (base_date - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
    for _ in range(n_customers)
]

# Build DataFrame
df = pd.DataFrame({
    'customer_id': customer_ids,
    'join_date': join_date_str,
    'last_interaction_date': last_interaction,
    'age': ages,
    'gender': genders,
    'city': cities,
    'state': state_col,
    'is_senior_citizen': is_senior_citizen,
    'has_partner': has_partner,
    'has_dependents': has_dependents,
    'plan': plans,
    'contract_type': contract_types,
    'payment_method': payment_methods,
    'paperless_billing': paperless_billing,
    'internet_service': internet_service,
    'multiple_lines': multiple_lines,
    'streaming_tv': streaming_tv,
    'streaming_movies': streaming_movies,
    'online_backup': online_backup,
    'device_protection': device_protection,
    'tech_support': tech_support,
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'avg_monthly_data_gb': avg_monthly_data_gb,
    'avg_call_minutes_per_month': avg_call_minutes,
    'support_calls_last_6months': support_calls_last_6m,
    'complaints_last_year': complaints_last_year,
    'network_downtime_hours_last_month': network_downtime_hours,
    'avg_satisfaction_rating': avg_rating,
    'payment_delays_last_year': payment_delays,
    'churn_probability': churn_prob.round(4),
    'churn': churn
})

# Save dataset in current project folder
df.to_csv('data/telecom_churn_dataset.csv', index=False)
print(f"Dataset created: {len(df)} rows, {len(df.columns)} columns")
print(f"Churn Rate: {df['churn'].mean():.2%}")
print(df.head())
