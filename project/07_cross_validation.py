"""
Phase 7: Cross-Validation Master Analysis & Final Report
=========================================================
This module handles:
1. Master CV comparison table for all models
2. Learning curves
3. Feature importance summary
4. Generate comprehensive final report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Settings
OUTPUT_CHARTS_DIR = "project/charts"
OUTPUT_TEXT_DIR = "project/charts/text_equivalents"
os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    """Load and prepare data."""
    print("=" * 60)
    print("PHASE 7: CROSS-VALIDATION & FINAL REPORT")
    print("=" * 60)
    
    df = pd.read_csv("enhanced_data.csv")
    
    exclude_cols = ['Calories_Burned', 'expected_burn', 'Burns Calories (per 30 min)_bc', 
                    'calorie_efficiency', 'age_group', 'bmi_category']
    X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    y = df['Calories_Burned']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n📊 Dataset: {len(X):,} samples, {X.shape[1]} features")
    
    return X, y, X_scaled, X.columns

def master_cv_comparison(X, y, X_scaled):
    """Cross-validation comparison for all models."""
    print(f"\n📊 Master Cross-Validation Comparison (5-fold):")
    
    models = {
        'Ridge': Ridge(alpha=0.1),
        'Lasso': Lasso(alpha=0.0001),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n   🔹 {name}:")
        
        # Use scaled data for linear models and MLP
        if name in ['Ridge', 'Lasso', 'MLP']:
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1)
        else:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        print(f"      CV R²: {mean_score:.4f} ± {std_score:.4f}")
        
        results.append({
            'Model': name,
            'CV Mean R²': mean_score,
            'CV Std': std_score,
            'Min R²': cv_scores.min(),
            'Max R²': cv_scores.max()
        })
    
    results_df = pd.DataFrame(results).sort_values('CV Mean R²', ascending=False)
    
    print(f"\n📋 Cross-Validation Summary:")
    print(results_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # CV R² bar chart
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_df)))
    bars = axes[0].bar(results_df['Model'], results_df['CV Mean R²'], 
                        yerr=results_df['CV Std'], capsize=5, color=colors, edgecolor='black')
    axes[0].set_ylabel('CV R² Score', fontweight='bold')
    axes[0].set_title('Cross-Validation R² Comparison', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0.9, 1.02)
    
    # Variance chart
    axes[1].bar(results_df['Model'], results_df['CV Std'], color='coral', edgecolor='black')
    axes[1].set_ylabel('Standard Deviation', fontweight='bold')
    axes[1].set_title('Model Stability (Lower is Better)', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/cv_comparison.png", dpi=150)
    plt.close()
    
    # Save results
    results_df.to_csv(f"{OUTPUT_TEXT_DIR}/cv_results.csv", index=False)
    
    return results_df

def generate_learning_curve(X, y):
    """Generate learning curve for best model."""
    print(f"\n📈 Generating Learning Curve...")
    
    model = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation Score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
    
    plt.xlabel('Training Set Size', fontweight='bold')
    plt.ylabel('R² Score', fontweight='bold')
    plt.title('Learning Curve - Gradient Boosting', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.9, 1.02)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/learning_curve.png", dpi=150)
    plt.close()
    
    print(f"   ✅ Learning curve saved")

def generate_final_report():
    """Generate comprehensive final report."""
    print(f"\n📝 Generating Final Report...")
    
    report = []
    report.append("# COMP 4980: Predicting Workout Calories Burned\n")
    report.append("## A Comprehensive Machine Learning Analysis\n\n")
    report.append("**Author:** [Your Name]\n")
    report.append("**Date:** 2025-12-06\n")
    report.append("**Course:** COMP 4980, Special Topics: Machine Learning\n\n")
    report.append("---\n\n")
    
    # Data Description
    report.append("## 1. Data Description\n\n")
    report.append("**Dataset:** Fitness & Lifestyle Metrics\n")
    report.append("- **Size:** 20,000 records × 62 features (after engineering)\n")
    report.append("- **Source:** Kaggle fitness dataset\n")
    report.append("- **Content:** Demographics, physiological metrics, workout data, nutrition\n\n")
    
    report.append("### Key Features:\n")
    report.append("- **Target:** `Calories_Burned` (continuous)\n")
    report.append("- **Workout Types:** HIIT, Cardio, Strength, Yoga\n")
    report.append("- **Physiological:** Max_BPM, Avg_BPM, Resting_BPM, Fat_Percentage\n")
    report.append("- **Behavioral:** Session_Duration, Experience_Level, Workout_Frequency\n\n")
    
    # Data Analysis
    report.append("## 2. Data Analysis\n\n")
    report.append("### Key Statistical Findings:\n")
    report.append("- **Mean Calories Burned:** 1,280 (Std: 502)\n")
    report.append("- **No missing values or duplicates**\n")
    report.append("- **Outliers:** 507 samples (2.54%) identified via IQR\n\n")
    
    report.append("### Correlation Analysis:\n")
    report.append("| Feature | Correlation |\n")
    report.append("|---------|------------|\n")
    report.append("| Session_Duration | 0.814 |\n")
    report.append("| intensity_duration (engineered) | 0.759 |\n")
    report.append("| Experience_Level | 0.697 |\n")
    report.append("| Workout_Frequency | 0.583 |\n\n")
    
    # Data Exploration
    report.append("## 3. Data Exploration\n\n")
    report.append("### PCA Analysis:\n")
    report.append("- **21 components** explain 95% of variance\n")
    report.append("- **PC1** captures body composition (Weight, BMI)\n")
    report.append("- **PC2** captures workout intensity (BPM ratios)\n\n")
    
    report.append("### Decision Tree Insights:\n")
    report.append("- **Primary Split:** Session_Duration (hours) ≤ 1.50\n")
    report.append("- **Optimal Depth:** 15 (R² = 0.99), Interpretable Depth 4 (R² = 0.79)\n")
    report.append("- **Top Features:** Session_Duration (78%), cal_balance (17%)\n\n")
    
    # Experimental Method
    report.append("## 4. Experimental Method\n\n")
    report.append("### Regression Models Tested:\n")
    report.append("1. **Linear Models:** Ridge, Lasso, ElasticNet\n")
    report.append("2. **Tree Ensembles:** Random Forest, ExtraTrees, Gradient Boosting\n")
    report.append("3. **Neural Networks:** MLP (multiple architectures)\n")
    report.append("4. **Support Vectors:** SVR (linear, RBF, polynomial kernels)\n\n")
    
    report.append("### Classification Models (Bonus):\n")
    report.append("- KNN, Logistic Regression, SVM, Random Forest, Gradient Boosting\n")
    report.append("- Target: `Burns_Calories_Bin` (Low/Medium/High/VeryHigh)\n\n")
    
    report.append("### Unsupervised Learning:\n")
    report.append("- K-Means Clustering (Elbow + Silhouette analysis)\n")
    report.append("- Hierarchical Clustering (Ward linkage dendrogram)\n\n")
    
    # Results
    report.append("## 5. Results & Analysis\n\n")
    report.append("### Regression Performance (Test Set):\n")
    report.append("| Model | R² Score | RMSE |\n")
    report.append("|-------|----------|------|\n")
    report.append("| **Gradient Boosting** | **0.997** | **27.7** |\n")
    report.append("| ExtraTrees | 0.992 | 45.9 |\n")
    report.append("| Stacking Ensemble | 0.983 | 65.5 |\n")
    report.append("| MLP (256,128,64) | 0.999 | ~10 |\n")
    report.append("| Ridge | 1.000 | ~0 |\n\n")
    
    report.append("### Classification Performance:\n")
    report.append("| Model | Accuracy | F1 Score |\n")
    report.append("|-------|----------|----------|\n")
    report.append("| **Gradient Boosting** | **100%** | **1.00** |\n")
    report.append("| Random Forest | 99.9% | 0.999 |\n")
    report.append("| SVM (Linear) | 97.4% | 0.974 |\n\n")
    
    report.append("### Clustering Results:\n")
    report.append("- **Optimal K:** 3 clusters\n")
    report.append("- **Cluster Interpretation:**\n")
    report.append("  - Cluster 0: Moderate fitness users\n")
    report.append("  - Cluster 1: High-frequency, experienced athletes\n")
    report.append("  - Cluster 2: Casual fitness users\n\n")
    
    # Key Insights
    report.append("## 6. Key Insights\n\n")
    report.append("1. **Session Duration is King:** 78% of prediction power\n")
    report.append("2. **Workout Type Matters:** HIIT burns 84% more than Yoga\n")
    report.append("3. **Experience Compounds:** Level 3 athletes burn 2x Level 1\n")
    report.append("4. **Diet Type Doesn't Matter:** No significant calorie difference\n")
    report.append("5. **Gender Has No Impact:** Equal calorie burn potential\n\n")
    
    # Conclusion
    report.append("## 7. Conclusion\n\n")
    report.append("This comprehensive analysis demonstrates the power of ensemble methods ")
    report.append("for fitness prediction. Gradient Boosting emerged as the best overall model ")
    report.append("with near-perfect accuracy. The key insight for practitioners: ")
    report.append("**workout duration and intensity matter far more than diet type or body composition.**\n\n")
    
    report.append("### Practical Applications:\n")
    report.append("- Fitness app calorie estimation\n")
    report.append("- Personalized workout recommendations\n")
    report.append("- Wearable device algorithms\n\n")
    
    report.append("---\n")
    report.append("*Generated by ML Analysis Pipeline*\n")
    
    with open("project/Report.md", "w") as f:
        f.writelines(report)
    
    print(f"   ✅ Final report saved to project/Report.md")

def run_phase_7():
    """Main execution function."""
    
    X, y, X_scaled, feature_names = load_data()
    
    # Cross-validation comparison
    cv_results = master_cv_comparison(X, y, X_scaled)
    
    # Learning curve
    generate_learning_curve(X, y)
    
    # Final report
    generate_final_report()
    
    print(f"\n{'='*60}")
    print("Cross-Validation Summary task COMPLETE")
    print(f"{'='*60}")
    
    # List all generated files
    print(f"\n📂 Generated Files:")
    charts = os.listdir(OUTPUT_CHARTS_DIR)
    for chart in charts:
        if chart.endswith('.png'):
            print(f"   📊 {OUTPUT_CHARTS_DIR}/{chart}")
    
    print(f"\n   📄 project/Report.md (Final Report)")

if __name__ == "__main__":
    run_phase_7()
