# 📡 Customer Churn Prediction Dashboard
## TeleVision Plus — Data Analytics Project

---

## 🎯 Project Overview

A complete end-to-end **Customer Churn Prediction** system for a Telecom/OTT company (TeleVision Plus). This project uses Machine Learning to predict which customers are likely to leave, and provides an interactive dashboard with retention strategies and churn probabilities.

---

## 📁 Project Structure

```
churn_project/
│
├── 📊 Data Files
│   ├── telecom_churn_dataset.csv      # Main dataset (5,000 customers)
│   ├── churn_predictions.csv          # ML predictions + risk segments
│   ├── model_metrics.csv              # Model performance comparison
│   ├── feature_importance.csv         # Top predictive features
│   └── eda_summary.csv               # EDA statistical summary
│
├── 🐍 Python Scripts
│   ├── generate_dataset.py            # Synthetic dataset generation
│   ├── eda_analysis.py                # Exploratory Data Analysis
│   └── train_model.py                 # ML model training
│
├── 📈 Visual Outputs
│   ├── eda_analysis.png              # EDA charts (11 visualizations)
│   └── model_evaluation.png          # Model performance charts
│
├── 🤖 Model Artifacts
│   ├── best_model.pkl                 # Trained Random Forest model
│   ├── scaler.pkl                     # StandardScaler
│   ├── label_encoders.pkl             # LabelEncoders for categories
│   └── model_info.json               # Model metadata
│
├── 🌐 Dashboard
│   └── dashboard.html                 # Interactive web dashboard
│
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_dataset.py
```

### 3. Run EDA
```bash
python eda_analysis.py
```

### 4. Train ML Models
```bash
python train_model.py
```

### 5. Open Dashboard
```bash
# Simply open dashboard.html in any web browser
open dashboard.html
```

---

## 📊 Dataset Description

| Feature | Type | Description |
|---------|------|-------------|
| customer_id | String | Unique customer identifier |
| age | Integer | Customer age (18–75) |
| gender | Category | Male / Female / Other |
| city | Category | 10 major Indian cities |
| plan | Category | Basic / Standard / Premium / Enterprise |
| contract_type | Category | Month-to-Month / One Year / Two Year |
| tenure_months | Integer | How long the customer has been with us |
| monthly_charges | Float | Monthly bill amount (₹) |
| total_charges | Float | Total amount paid so far (₹) |
| internet_service | Category | Fiber Optic / Cable / DSL / No |
| avg_satisfaction_rating | Float | 1–5 star rating |
| support_calls_last_6months | Integer | Number of support calls |
| complaints_last_year | Integer | Number of complaints filed |
| payment_delays_last_year | Integer | Number of late payments |
| network_downtime_hours | Float | Network outage hours last month |
| churn | Binary | 0 = Retained, 1 = Churned (Target) |

---

## 🤖 ML Models Trained

| Model | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|----------|
| Logistic Regression | 63.0% | 0.646 | 0.465 |
| **Random Forest** ⭐ | **62.4%** | **0.648** | **0.405** |
| Gradient Boosting | 60.9% | 0.632 | 0.481 |

**Best Model:** Random Forest (highest ROC-AUC)

---

## 🎯 Risk Segments

| Segment | Churn Prob | Count | Action |
|---------|-----------|-------|--------|
| 🟢 Low Risk | < 30% | ~2,240 | Loyalty program |
| 🟡 Medium Risk | 30–50% | ~1,482 | Proactive outreach |
| 🟠 High Risk | 50–70% | ~891 | Immediate call + offer |
| 🔴 Critical Risk | ≥ 70% | ~387 | VIP retention intervention |

---

## 💡 Key Insights

1. **Contract Type is the #1 factor** — Month-to-Month customers churn 3.5x more than 2-Year contracts
2. **New customers are most vulnerable** — 65% churn in first 6 months vs 18% after 3 years
3. **Satisfaction rating drives churn** — Every 1-point drop increases churn risk by ~18%
4. **Support calls signal frustration** — 3+ calls = 61% churn rate vs 28% for 0 calls
5. **Revenue at risk** — ~₹8.4 Lakhs/month MRR from critical + high risk segments

---

## 📈 Dashboard Features

- **Overview Tab**: KPI cards, churn distribution charts, city/plan/tenure analysis
- **Customer Risk Table**: Searchable/filterable table with churn probability bars
- **ML Model Metrics**: ROC curves, confusion matrix, feature importance
- **Retention Strategies**: 4-tier strategy framework with actionable recommendations
- **Insights Tab**: Business insights and priority action items

---

## 👨‍💻 Tech Stack

- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **ML**: Logistic Regression, Random Forest, Gradient Boosting
- **Dashboard**: HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Data**: Synthetic dataset (real-world patterns)

---

## 📌 Author Notes

This is a **dummy dataset** created for educational/portfolio purposes. The patterns are designed to mimic real-world telecom churn behavior based on:
- Industry-standard churn drivers
- Realistic Indian telecom pricing (₹)
- Authentic Indian cities and geography
- Plausible customer behavior distributions

---


