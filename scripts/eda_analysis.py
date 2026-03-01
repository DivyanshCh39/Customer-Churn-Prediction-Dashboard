"""
Exploratory Data Analysis (EDA)
Customer Churn Prediction - TeleVision Plus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_palette("husl")

print("=" * 60)
print("  EDA - Customer Churn Prediction")
print("  TeleVision Plus Analytics")
print("=" * 60)

df = pd.read_csv('data/telecom_churn_dataset.csv')

print("\n📊 Dataset Overview")
print(f"   Shape: {df.shape}")
print(f"   Churn Rate: {df['churn'].mean():.2%}")
print(f"\n   Data Types:\n{df.dtypes.value_counts()}")
print(f"\n   Missing Values: {df.isnull().sum().sum()}")

print("\n📊 Churn Distribution:")
print(df['churn'].value_counts())

print("\n📊 Churn by Contract Type:")
print(df.groupby('contract_type')['churn'].mean().sort_values(ascending=False).map(lambda x: f"{x:.2%}"))

print("\n📊 Churn by Plan:")
print(df.groupby('plan')['churn'].mean().sort_values(ascending=False).map(lambda x: f"{x:.2%}"))

print("\n📊 Average Metrics by Churn Status:")
num_cols = ['tenure_months','monthly_charges','avg_satisfaction_rating',
            'support_calls_last_6months','complaints_last_year',
            'payment_delays_last_year','network_downtime_hours_last_month']
print(df.groupby('churn')[num_cols].mean().round(2).T)

# ── Plots ──
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Customer Churn EDA — TeleVision Plus\nComprehensive Analysis', 
             fontsize=18, fontweight='bold', color='white', y=0.98)

colors_churn = ['#4CAF50', '#F44336']
DARK_BG = '#161b22'

# 1. Churn distribution
ax1 = fig.add_subplot(gs[0, 0])
churn_counts = df['churn'].value_counts()
bars = ax1.bar(['Retained', 'Churned'], churn_counts.values, color=colors_churn, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, churn_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val}\n({val/len(df):.1%})', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_title('Churn Distribution', fontweight='bold')
ax1.set_facecolor(DARK_BG)
ax1.set_ylabel('Count')

# 2. Churn by contract
ax2 = fig.add_subplot(gs[0, 1])
ct_churn = df.groupby('contract_type')['churn'].mean().sort_values(ascending=False)
bars = ax2.bar(ct_churn.index, ct_churn.values * 100,
               color=['#F44336','#FF9800','#4CAF50'], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, ct_churn.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_title('Churn Rate by Contract Type', fontweight='bold')
ax2.set_facecolor(DARK_BG)
ax2.set_ylabel('Churn Rate (%)')
ax2.set_xticklabels(ct_churn.index, rotation=15, ha='right')

# 3. Churn by plan
ax3 = fig.add_subplot(gs[0, 2])
plan_churn = df.groupby('plan')['churn'].mean().sort_values(ascending=False)
bars = ax3.bar(plan_churn.index, plan_churn.values * 100,
               color=['#F44336','#FF9800','#2196F3','#4CAF50'], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, plan_churn.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax3.set_title('Churn Rate by Plan', fontweight='bold')
ax3.set_facecolor(DARK_BG)
ax3.set_ylabel('Churn Rate (%)')

# 4. Tenure distribution
ax4 = fig.add_subplot(gs[1, 0])
for churn_val, color, label in [(0, '#4CAF50', 'Retained'), (1, '#F44336', 'Churned')]:
    ax4.hist(df[df['churn']==churn_val]['tenure_months'], bins=30, alpha=0.6,
             color=color, label=label, density=True)
ax4.set_title('Tenure Distribution', fontweight='bold')
ax4.set_facecolor(DARK_BG)
ax4.set_xlabel('Tenure (Months)')
ax4.legend()

# 5. Monthly charges boxplot
ax5 = fig.add_subplot(gs[1, 1])
data_retained = df[df['churn']==0]['monthly_charges']
data_churned = df[df['churn']==1]['monthly_charges']
bp = ax5.boxplot([data_retained, data_churned], labels=['Retained', 'Churned'],
                  patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], colors_churn):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_title('Monthly Charges Distribution', fontweight='bold')
ax5.set_facecolor(DARK_BG)
ax5.set_ylabel('Monthly Charges (₹)')

# 6. Satisfaction rating
ax6 = fig.add_subplot(gs[1, 2])
for churn_val, color, label in [(0, '#4CAF50', 'Retained'), (1, '#F44336', 'Churned')]:
    ax6.hist(df[df['churn']==churn_val]['avg_satisfaction_rating'], bins=20, alpha=0.6,
             color=color, label=label, density=True)
ax6.set_title('Satisfaction Rating Distribution', fontweight='bold')
ax6.set_facecolor(DARK_BG)
ax6.set_xlabel('Rating (1–5)')
ax6.legend()

# 7. Correlation heatmap
ax7 = fig.add_subplot(gs[2, :2])
corr_cols = ['churn','tenure_months','monthly_charges','total_charges',
             'support_calls_last_6months','complaints_last_year',
             'avg_satisfaction_rating','payment_delays_last_year',
             'network_downtime_hours_last_month','avg_monthly_data_gb']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax7, linewidths=0.5, annot_kws={'size': 9})
ax7.set_title('Feature Correlation Heatmap', fontweight='bold')

# 8. Internet service churn
ax8 = fig.add_subplot(gs[2, 2])
is_churn = df.groupby('internet_service')['churn'].mean().sort_values(ascending=False)
colors_is = ['#F44336','#FF9800','#FF9800','#4CAF50']
bars = ax8.bar(range(len(is_churn)), is_churn.values * 100, color=colors_is[:len(is_churn)],
               edgecolor='white', linewidth=0.5)
ax8.set_xticks(range(len(is_churn)))
ax8.set_xticklabels(is_churn.index, rotation=15, ha='right', fontsize=9)
ax8.set_title('Churn by Internet Service', fontweight='bold')
ax8.set_facecolor(DARK_BG)
ax8.set_ylabel('Churn Rate (%)')

# 9. Churn by city
ax9 = fig.add_subplot(gs[3, 0])
city_churn = df.groupby('city')['churn'].mean().sort_values(ascending=False).head(8)
colors_city = plt.cm.RdYlGn_r(np.linspace(0, 1, len(city_churn)))
bars = ax9.barh(city_churn.index[::-1], city_churn.values[::-1] * 100,
                color=colors_city, edgecolor='white', linewidth=0.3)
ax9.set_title('Churn Rate by City', fontweight='bold')
ax9.set_facecolor(DARK_BG)
ax9.set_xlabel('Churn Rate (%)')

# 10. Support calls
ax10 = fig.add_subplot(gs[3, 1])
sc_churn = df.groupby('support_calls_last_6months')['churn'].mean().head(8)
ax10.plot(sc_churn.index, sc_churn.values * 100, 'o-', color='#FF9800',
          linewidth=2.5, markersize=8, markerfacecolor='white')
ax10.fill_between(sc_churn.index, sc_churn.values * 100, alpha=0.2, color='#FF9800')
ax10.set_title('Churn vs Support Calls', fontweight='bold')
ax10.set_facecolor(DARK_BG)
ax10.set_xlabel('Support Calls (last 6 months)')
ax10.set_ylabel('Churn Rate (%)')
ax10.grid(True, alpha=0.2)

# 11. Payment delays
ax11 = fig.add_subplot(gs[3, 2])
pd_churn = df.groupby('payment_delays_last_year')['churn'].mean().head(7)
bars = ax11.bar(pd_churn.index, pd_churn.values * 100,
                color=plt.cm.Reds(np.linspace(0.4, 0.9, len(pd_churn))),
                edgecolor='white', linewidth=0.3)
ax11.set_title('Churn vs Payment Delays', fontweight='bold')
ax11.set_facecolor(DARK_BG)
ax11.set_xlabel('Payment Delays (last year)')
ax11.set_ylabel('Churn Rate (%)')

plt.savefig('eda_analysis.png', dpi=130, bbox_inches='tight', facecolor='#0d1117')
print("\n✅ EDA plots saved: eda_analysis.png")

# Summary stats
summary = df.groupby('churn').agg(
    count=('customer_id', 'count'),
    avg_tenure=('tenure_months', 'mean'),
    avg_monthly_charges=('monthly_charges', 'mean'),
    avg_satisfaction=('avg_satisfaction_rating', 'mean'),
    avg_complaints=('complaints_last_year', 'mean'),
    avg_support_calls=('support_calls_last_6months', 'mean')
).round(2)
summary.to_csv('eda_summary.csv')
print("✅ EDA summary saved: eda_summary.csv")
print("\n" + summary.to_string())
print("\n✅ EDA Complete!")
