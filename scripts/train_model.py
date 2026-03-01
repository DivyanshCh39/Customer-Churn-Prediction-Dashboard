"""
Customer Churn Prediction - ML Model Training
Company: TeleVision Plus (Dummy Dataset)
Models: Logistic Regression, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("  Customer Churn Prediction - Model Training")
print("  TeleVision Plus - Data Analytics Project")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('data/telecom_churn_dataset.csv')
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Churn Rate: {df['churn'].mean():.2%}")

# ── 2. Feature Engineering ────────────────────────────────────
df_model = df.copy()

# Drop non-feature columns
drop_cols = ['customer_id', 'join_date', 'last_interaction_date', 'churn_probability', 'churn']
X = df_model.drop(columns=drop_cols)
y = df_model['churn']

# Encode categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Scale numerical features
scaler = StandardScaler()
num_cols = X.select_dtypes(include=np.number).columns.tolist()
X_scaled = X.copy()
X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

print(f"\n✅ Features prepared: {X.shape[1]} features")

# ── 3. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# ── 4. Train Models ───────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
}

results = {}
print("\n📊 Training Models...")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"\n🤖 {name}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"   CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. Best Model ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_name]['model']
print(f"\n🏆 Best Model: {best_name} (AUC: {results[best_name]['roc_auc']:.4f})")

# ── 6. Feature Importance ─────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_imp = feat_imp.sort_values(ascending=False).head(15)
    feat_imp.to_csv('feature_importance.csv')
    print("\n✅ Feature importance saved")

# ── 7. Save Predictions on Full Dataset ───────────────────────
X_full = X_scaled
df['predicted_churn'] = best_model.predict(X_full)
df['churn_probability_ml'] = best_model.predict_proba(X_full)[:, 1].round(4)

# Risk segments
df['risk_segment'] = pd.cut(
    df['churn_probability_ml'],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
)

# Retention strategy
def get_retention_strategy(row):
    if row['churn_probability_ml'] < 0.3:
        return "Maintain Engagement - Send loyalty rewards"
    elif row['churn_probability_ml'] < 0.5:
        return "Proactive Outreach - Offer plan upgrade discount"
    elif row['churn_probability_ml'] < 0.7:
        return "Immediate Action - Call center follow-up + special offer"
    else:
        return "URGENT - Retention team intervention + VIP offer"

df['retention_strategy'] = df.apply(get_retention_strategy, axis=1)
df.to_csv('churn_predictions.csv', index=False)
print(f"✅ Predictions saved: churn_predictions.csv")

# ── 8. Model Metrics Summary ──────────────────────────────────
metrics_summary = []
for name, res in results.items():
    report = classification_report(y_test, res['y_pred'], output_dict=True)
    metrics_summary.append({
        'Model': name,
        'Accuracy': round(res['accuracy'], 4),
        'F1_Score': round(res['f1_score'], 4),
        'ROC_AUC': round(res['roc_auc'], 4),
        'CV_AUC_Mean': round(res['cv_mean'], 4),
        'CV_AUC_Std': round(res['cv_std'], 4),
        'Precision_Churn': round(report['1']['precision'], 4),
        'Recall_Churn': round(report['1']['recall'], 4)
    })

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('model_metrics.csv', index=False)
print("✅ Model metrics saved: model_metrics.csv")

# ── 9. Generate Plots ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Churn Prediction - Model Evaluation\nTeleVision Plus', fontsize=16, fontweight='bold')

colors = ['#2196F3', '#4CAF50', '#FF5722']

# ROC Curves
ax = axes[0, 0]
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - All Models')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# Confusion Matrix - Best Model
ax = axes[0, 1]
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax.set_title(f'Confusion Matrix\n{best_name}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# Feature Importance
ax = axes[0, 2]
if hasattr(best_model, 'feature_importances_'):
    feat_imp_plot = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)
    feat_imp_plot.plot(kind='barh', ax=ax, color='#2196F3')
    ax.set_title(f'Feature Importance\n{best_name}')
    ax.set_xlabel('Importance Score')
    ax.grid(True, alpha=0.3)

# Model Comparison
ax = axes[1, 0]
model_names = [r['Model'] for r in metrics_summary]
auc_scores = [r['ROC_AUC'] for r in metrics_summary]
bars = ax.bar(model_names, auc_scores, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('ROC-AUC Score')
ax.set_title('Model Comparison - ROC AUC')
ax.set_ylim(0.5, 1.0)
for bar, score in zip(bars, auc_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=15)

# Churn Distribution by Risk Segment
ax = axes[1, 1]
risk_counts = df['risk_segment'].value_counts()
colors_risk = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0']
wedges, texts, autotexts = ax.pie(risk_counts, labels=risk_counts.index,
                                   autopct='%1.1f%%', colors=colors_risk,
                                   startangle=90, pctdistance=0.85)
ax.set_title('Customer Distribution\nby Risk Segment')

# Churn Probability Distribution
ax = axes[1, 2]
ax.hist(df[df['churn'] == 0]['churn_probability_ml'], bins=40, alpha=0.6,
        color='#4CAF50', label='No Churn', density=True)
ax.hist(df[df['churn'] == 1]['churn_probability_ml'], bins=40, alpha=0.6,
        color='#F44336', label='Churned', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
ax.set_xlabel('Predicted Churn Probability')
ax.set_ylabel('Density')
ax.set_title('Churn Probability Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
print("✅ Plots saved: model_evaluation.png")
plt.close()

# ── 10. Save model artifacts ──────────────────────────────────
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

model_info = {
    'best_model': best_name,
    'features': list(X.columns),
    'cat_cols': cat_cols,
    'num_cols': num_cols,
    'metrics': {name: {'accuracy': res['accuracy'], 'roc_auc': res['roc_auc'], 'f1_score': res['f1_score']}
                for name, res in results.items()},
    'trained_on': str(pd.Timestamp.now())
}
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n✅ Model artifacts saved (best_model.pkl, scaler.pkl, label_encoders.pkl, model_info.json)")
print("\n" + "=" * 60)
print("  Model Training Complete!")
print(f"  Best Model: {best_name}")
print(f"  ROC-AUC: {results[best_name]['roc_auc']:.4f}")
print("=" * 60)
