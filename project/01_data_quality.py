"""
Phase 1 Enhanced: Data Quality Report & Feature Engineering
============================================================
This module handles:
1. Data loading and initial inspection
2. Outlier detection using IQR method
3. Feature engineering (new derived features)
4. Comprehensive correlation analysis
5. Data quality report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Settings
OUTPUT_CHARTS_DIR = "project/charts"
OUTPUT_TEXT_DIR = "project/charts/text_equivalents"
os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_inspect_data():
    """Load dataset and perform initial inspection."""
    print("=" * 60)
    print("PHASE 1 ENHANCED: DATA QUALITY & FEATURE ENGINEERING")
    print("=" * 60)
    
    df = pd.read_csv("dataset.csv")
    
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Data types summary
    print(f"\n📋 Data Types:")
    print(f"   Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    if missing.sum() > 0:
        print(f"\n⚠️  Missing Values Found:")
        for col in missing[missing > 0].index:
            print(f"   {col}: {missing[col]} ({missing_pct[col]}%)")
    else:
        print(f"\n✅ No Missing Values")
    
    # Duplicates
    dupes = df.duplicated().sum()
    print(f"{'⚠️' if dupes > 0 else '✅'} Duplicates: {dupes}")
    
    return df

def detect_outliers(df, column='Calories_Burned'):
    """Detect outliers using IQR method."""
    print(f"\n🔍 Outlier Detection (IQR Method) for '{column}':")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   Outliers Found: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    # Visualize outliers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    axes[0].boxplot(df[column], vert=True)
    axes[0].set_title(f'Box Plot: {column}', fontweight='bold', fontsize=12)
    axes[0].set_ylabel(column)
    axes[0].axhline(y=lower_bound, color='r', linestyle='--', alpha=0.7, label='Lower Bound')
    axes[0].axhline(y=upper_bound, color='r', linestyle='--', alpha=0.7, label='Upper Bound')
    axes[0].legend()
    
    # Distribution with outliers highlighted
    axes[1].hist(df[column], bins=50, alpha=0.7, label='All Data', edgecolor='black')
    axes[1].axvline(x=lower_bound, color='r', linestyle='--', label='Outlier Bounds')
    axes[1].axvline(x=upper_bound, color='r', linestyle='--')
    axes[1].set_title(f'Distribution with Outlier Bounds', fontweight='bold', fontsize=12)
    axes[1].set_xlabel(column)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/outlier_analysis.png", dpi=150)
    plt.close()
    
    return outliers, lower_bound, upper_bound

def engineer_features(df):
    """Create new engineered features based on domain knowledge."""
    print(f"\n🔧 Feature Engineering:")
    
    df_eng = df.copy()
    
    # 1. Intensity-Duration Interaction (validated: 0.76 correlation)
    df_eng['intensity_duration'] = df_eng['Avg_BPM'] * df_eng['Session_Duration (hours)']
    print(f"   ✓ intensity_duration = Avg_BPM × Session_Duration")
    
    # 2. BPM Intensity Ratio (effort level)
    df_eng['bpm_intensity'] = df_eng['Avg_BPM'] / df_eng['Max_BPM']
    print(f"   ✓ bpm_intensity = Avg_BPM / Max_BPM")
    
    # 3. Heart Rate Reserve Usage
    df_eng['hrr_usage'] = (df_eng['Avg_BPM'] - df_eng['Resting_BPM']) / (df_eng['Max_BPM'] - df_eng['Resting_BPM'])
    print(f"   ✓ hrr_usage = (Avg - Resting) / (Max - Resting)")
    
    # 4. Age Groups
    df_eng['age_group'] = pd.cut(df_eng['Age'], 
                                  bins=[0, 25, 35, 45, 55, 100],
                                  labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    print(f"   ✓ age_group = categorical age bins")
    
    # 5. BMI Category (WHO Classification)
    def bmi_category(bmi):
        if bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal'
        elif bmi < 30: return 'Overweight'
        else: return 'Obese'
    df_eng['bmi_category'] = df_eng['BMI'].apply(bmi_category)
    print(f"   ✓ bmi_category = WHO BMI classification")
    
    # 6. Macro Ratio (Protein-centric)
    df_eng['protein_ratio'] = df_eng['Proteins'] / (df_eng['Carbs'] + df_eng['Fats'] + 1)
    print(f"   ✓ protein_ratio = Proteins / (Carbs + Fats)")
    
    # 7. Calorie Efficiency (calories burned per minute)
    df_eng['calorie_efficiency'] = df_eng['Calories_Burned'] / (df_eng['Session_Duration (hours)'] * 60)
    print(f"   ✓ calorie_efficiency = Calories / Minutes")
    
    # 8. Experience Score (normalized)
    df_eng['experience_score'] = (df_eng['Experience_Level'] - df_eng['Experience_Level'].min()) / \
                                  (df_eng['Experience_Level'].max() - df_eng['Experience_Level'].min())
    print(f"   ✓ experience_score = normalized experience level")
    
    # Validate new features
    print(f"\n📈 New Feature Correlations with Calories_Burned:")
    new_features = ['intensity_duration', 'bpm_intensity', 'hrr_usage', 'protein_ratio', 'calorie_efficiency']
    for feat in new_features:
        corr = df_eng[feat].corr(df_eng['Calories_Burned'])
        print(f"   {feat}: {corr:.4f}")
    
    return df_eng

def create_correlation_analysis(df):
    """Create comprehensive correlation analysis with visualizations."""
    print(f"\n📊 Correlation Analysis:")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Target correlations
    target_corr = corr_matrix['Calories_Burned'].drop('Calories_Burned').sort_values(ascending=False)
    
    print(f"\n   Top 10 Positive Correlations:")
    for feat, corr in target_corr.head(10).items():
        print(f"      {feat}: {corr:.4f}")
    
    print(f"\n   Top 5 Negative Correlations:")
    for feat, corr in target_corr.tail(5).items():
        print(f"      {feat}: {corr:.4f}")
    
    # Save correlation summary
    with open(f"{OUTPUT_TEXT_DIR}/correlation_analysis.txt", "w") as f:
        f.write("CORRELATION ANALYSIS WITH CALORIES_BURNED\n")
        f.write("=" * 50 + "\n\n")
        f.write(target_corr.to_string())
    
    # 1. Clustered Heatmap (hierarchical)
    plt.figure(figsize=(16, 14))
    
    # Select top 20 features for readability
    top_features = ['Calories_Burned'] + list(target_corr.head(15).index)
    sub_corr = df[top_features].corr()
    
    sns.heatmap(sub_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, annot_kws={'size': 8})
    plt.title('Correlation Heatmap: Top 15 Features vs Calories_Burned', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/correlation_heatmap_detailed.png", dpi=150)
    plt.close()
    
    # 2. Top Features Bar Chart
    plt.figure(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in target_corr.head(15)]
    target_corr.head(15).plot(kind='barh', color=colors, edgecolor='black')
    plt.xlabel('Correlation with Calories_Burned', fontweight='bold')
    plt.title('Top 15 Feature Correlations', fontweight='bold', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/top_correlations_bar.png", dpi=150)
    plt.close()
    
    return corr_matrix, target_corr

def generate_data_quality_report(df, outliers, target_corr):
    """Generate a comprehensive data quality report."""
    print(f"\n📝 Generating Data Quality Report...")
    
    report = []
    report.append("# Data Quality Report\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    
    report.append("## Dataset Overview\n")
    report.append(f"- **Rows:** {len(df):,}\n")
    report.append(f"- **Columns:** {len(df.columns)}\n")
    report.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n\n")
    
    report.append("## Data Types\n")
    report.append(f"- Numeric: {len(df.select_dtypes(include=[np.number]).columns)}\n")
    report.append(f"- Categorical: {len(df.select_dtypes(include=['object']).columns)}\n\n")
    
    report.append("## Target Variable (Calories_Burned)\n")
    stats = df['Calories_Burned'].describe()
    report.append(f"- Mean: {stats['mean']:.2f}\n")
    report.append(f"- Std: {stats['std']:.2f}\n")
    report.append(f"- Min: {stats['min']:.2f}\n")
    report.append(f"- Max: {stats['max']:.2f}\n")
    report.append(f"- Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)\n\n")
    
    report.append("## Top 10 Correlated Features\n")
    for i, (feat, corr) in enumerate(target_corr.head(10).items(), 1):
        report.append(f"{i}. **{feat}**: {corr:.4f}\n")
    
    report.append("\n## Engineered Features\n")
    report.append("- `intensity_duration`: BPM × Duration interaction\n")
    report.append("- `bpm_intensity`: Effort ratio (Avg/Max BPM)\n")
    report.append("- `hrr_usage`: Heart Rate Reserve utilization\n")
    report.append("- `age_group`: Categorical age bins\n")
    report.append("- `bmi_category`: WHO BMI classification\n")
    report.append("- `protein_ratio`: Protein-centric macro balance\n")
    report.append("- `calorie_efficiency`: Burn rate per minute\n")
    
    with open(f"{OUTPUT_TEXT_DIR}/data_quality_report.md", "w") as f:
        f.writelines(report)
    
    print(f"   ✅ Report saved to {OUTPUT_TEXT_DIR}/data_quality_report.md")

def run_phase_1_enhanced():
    """Main execution function for Phase 1 Enhanced."""
    
    # Step 1: Load and inspect
    df = load_and_inspect_data()
    
    # Step 2: Detect outliers
    outliers, lb, ub = detect_outliers(df)
    
    # Step 3: Feature engineering
    df_enhanced = engineer_features(df)
    
    # Step 4: Correlation analysis
    corr_matrix, target_corr = create_correlation_analysis(df_enhanced)
    
    # Step 5: Generate report
    generate_data_quality_report(df_enhanced, outliers, target_corr)
    
    # Step 6: Save enhanced dataset
    df_enhanced.to_csv("enhanced_data.csv", index=False)
    print(f"\n💾 Enhanced dataset saved to 'enhanced_data.csv'")
    
    print(f"\n{'='*60}")
    print("Data Quality & Feature Engineering task COMPLETE")
    print(f"{'='*60}")
    
    return df_enhanced

if __name__ == "__main__":
    run_phase_1_enhanced()
