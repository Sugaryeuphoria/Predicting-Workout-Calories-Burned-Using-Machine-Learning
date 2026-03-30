"""
Phase 3 Enhanced: PCA Deep Dive & Decision Tree Exploration
============================================================
This module handles:
1. Principal Component Analysis with biplot
2. Loadings interpretation
3. Variance explanation analysis
4. Decision Tree visualization suite
5. Tree depth optimization
6. Pruning experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
import os

# Settings
OUTPUT_CHARTS_DIR = "project/charts"
OUTPUT_TEXT_DIR = "project/charts/text_equivalents"
os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    """Load enhanced dataset."""
    print("=" * 60)
    print("PHASE 3 ENHANCED: PCA & DECISION TREE EXPLORATION")
    print("=" * 60)
    
    df = pd.read_csv("enhanced_data.csv")
    print(f"\n📊 Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def perform_pca_analysis(df):
    """Comprehensive PCA analysis with biplot and loadings."""
    print(f"\n🔬 Principal Component Analysis:")
    
    # Prepare numeric features (exclude target and leaky features)
    exclude_cols = ['Calories_Burned', 'expected_burn', 'Burns Calories (per 30 min)_bc', 
                    'calorie_efficiency', 'age_group', 'bmi_category']
    numeric_df = df.select_dtypes(include=[np.number])
    X = numeric_df.drop(columns=[c for c in exclude_cols if c in numeric_df.columns], errors='ignore')
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Variance explanation
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Find components for 95% variance
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_80 = np.argmax(cumulative_var >= 0.80) + 1
    
    print(f"\n   Variance Explanation:")
    print(f"   - 80% variance: {n_80} components")
    print(f"   - 90% variance: {n_90} components")
    print(f"   - 95% variance: {n_95} components")
    print(f"   - Total features: {X.shape[1]}")
    
    # Individual component variance (first 5)
    print(f"\n   Component-wise Variance (Top 5):")
    for i in range(min(5, len(explained_var))):
        print(f"   PC{i+1}: {explained_var[i]*100:.2f}% (cumulative: {cumulative_var[i]*100:.2f}%)")
    
    # === VISUALIZATION 1: Scree Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    components_to_plot = min(20, len(explained_var))
    axes[0].bar(range(1, components_to_plot + 1), explained_var[:components_to_plot] * 100, 
                alpha=0.7, color='steelblue', edgecolor='black', label='Individual')
    axes[0].plot(range(1, components_to_plot + 1), cumulative_var[:components_to_plot] * 100, 
                 'ro-', linewidth=2, markersize=6, label='Cumulative')
    axes[0].axhline(y=95, color='green', linestyle='--', label='95% threshold')
    axes[0].axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    axes[0].set_xlabel('Principal Component', fontweight='bold')
    axes[0].set_ylabel('Variance Explained (%)', fontweight='bold')
    axes[0].set_title('PCA Scree Plot', fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].set_xticks(range(1, components_to_plot + 1))
    
    # Cumulative variance curve
    axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, 'b-', linewidth=2)
    axes[1].axhline(y=95, color='green', linestyle='--', label=f'95% at PC{n_95}')
    axes[1].axvline(x=n_95, color='green', linestyle='--', alpha=0.5)
    axes[1].scatter([n_95], [95], color='green', s=100, zorder=5)
    axes[1].set_xlabel('Number of Components', fontweight='bold')
    axes[1].set_ylabel('Cumulative Variance (%)', fontweight='bold')
    axes[1].set_title('Cumulative Variance Explained', fontweight='bold', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/pca_scree_enhanced.png", dpi=150)
    plt.close()
    
    # === VISUALIZATION 2: Biplot ===
    print(f"\n   Creating Biplot (PC1 vs PC2)...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by Workout Type
    if 'Workout_Type' in df.columns:
        le = LabelEncoder()
        colors = le.fit_transform(df['Workout_Type'])
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='Set1', 
                            alpha=0.3, s=10)
        
        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=plt.cm.Set1(i/4), markersize=10, label=wt) 
                   for i, wt in enumerate(le.classes_)]
        ax.legend(handles=handles, title='Workout Type', loc='upper right')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10)
    
    # Add feature vectors (arrows)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    scale = 5  # Scale factor for arrows
    
    # Only plot top 10 most important features
    feature_importance = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_indices = np.argsort(feature_importance)[-10:]
    
    for i in top_indices:
        ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale, 
                head_width=0.15, head_length=0.1, fc='red', ec='red', alpha=0.7)
        ax.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15, 
                X.columns[i], fontsize=8, ha='center', va='center', fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontweight='bold')
    ax.set_title('PCA Biplot with Top 10 Feature Loadings', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/pca_biplot.png", dpi=150)
    plt.close()
    
    # === LOADINGS TABLE ===
    print(f"\n   Top Feature Loadings:")
    loadings_df = pd.DataFrame(pca.components_[:3].T, 
                                columns=['PC1', 'PC2', 'PC3'], 
                                index=X.columns)
    loadings_df['Importance'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)
    loadings_df = loadings_df.sort_values('Importance', ascending=False)
    
    print(f"\n   PC1 (Top 5 contributors):")
    for feat in loadings_df.nlargest(5, 'PC1').index:
        print(f"      {feat}: {loadings_df.loc[feat, 'PC1']:.4f}")
    
    print(f"\n   PC2 (Top 5 contributors):")
    for feat in loadings_df.nlargest(5, 'PC2').index:
        print(f"      {feat}: {loadings_df.loc[feat, 'PC2']:.4f}")
    
    # Save loadings
    loadings_df.to_csv(f"{OUTPUT_TEXT_DIR}/pca_loadings.csv")
    
    return pca, X, n_95

def decision_tree_analysis(df):
    """Comprehensive decision tree analysis."""
    print(f"\n🌳 Decision Tree Analysis:")
    
    # Prepare data
    exclude_cols = ['Calories_Burned', 'expected_burn', 'Burns Calories (per 30 min)_bc', 
                    'calorie_efficiency', 'age_group', 'bmi_category']
    X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    y = df['Calories_Burned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # === Test different depths ===
    print(f"\n   Testing Tree Depths (1-15):")
    depths = range(1, 16)
    train_scores = []
    test_scores = []
    
    for depth in depths:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_scores.append(dt.score(X_train, y_train))
        test_scores.append(dt.score(X_test, y_test))
    
    # Find optimal depth (best test score)
    optimal_depth = depths[np.argmax(test_scores)]
    print(f"   Optimal Depth: {optimal_depth} (Test R² = {max(test_scores):.4f})")
    
    # === VISUALIZATION: Depth vs Score ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(depths, train_scores, 'b-o', label='Train R²', linewidth=2, markersize=6)
    axes[0].plot(depths, test_scores, 'r-o', label='Test R²', linewidth=2, markersize=6)
    axes[0].axvline(x=optimal_depth, color='green', linestyle='--', label=f'Optimal Depth={optimal_depth}')
    axes[0].set_xlabel('Tree Depth', fontweight='bold')
    axes[0].set_ylabel('R² Score', fontweight='bold')
    axes[0].set_title('Tree Depth vs Performance (Bias-Variance Trade-off)', fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].set_xticks(depths)
    
    # Gap between train and test (overfitting indicator)
    gap = np.array(train_scores) - np.array(test_scores)
    axes[1].bar(depths, gap, color='orange', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Tree Depth', fontweight='bold')
    axes[1].set_ylabel('Train - Test R² (Overfitting Gap)', fontweight='bold')
    axes[1].set_title('Overfitting Analysis by Depth', fontweight='bold', fontsize=12)
    axes[1].axhline(y=0.05, color='red', linestyle='--', label='Acceptable Gap (0.05)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/decision_tree_depth_analysis.png", dpi=150)
    plt.close()
    
    # === Train final tree at depth 4 (interpretable) ===
    print(f"\n   Training Interpretable Tree (Depth 4)...")
    dt_final = DecisionTreeRegressor(max_depth=4, random_state=42)
    dt_final.fit(X_train, y_train)
    
    print(f"   Train R²: {dt_final.score(X_train, y_train):.4f}")
    print(f"   Test R²: {dt_final.score(X_test, y_test):.4f}")
    
    # === VISUALIZATION: Tree Diagram ===
    plt.figure(figsize=(24, 12))
    plot_tree(dt_final, feature_names=X.columns, filled=True, rounded=True, 
              fontsize=8, proportion=True, precision=1)
    plt.title('Decision Tree (Depth 4) - Calories Prediction', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/decision_tree_depth4.png", dpi=150)
    plt.close()
    
    # === Text Rules ===
    tree_rules = export_text(dt_final, feature_names=list(X.columns), max_depth=4)
    with open(f"{OUTPUT_TEXT_DIR}/decision_tree_rules.txt", "w") as f:
        f.write("DECISION TREE RULES (Depth 4)\n")
        f.write("=" * 50 + "\n\n")
        f.write(tree_rules)
    
    # === Feature Importance from Tree ===
    print(f"\n   Feature Importance (Gini):")
    importance = pd.Series(dt_final.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feat in importance.head(5).index:
        print(f"      {feat}: {importance[feat]:.4f}")
    
    # Feature importance bar chart
    plt.figure(figsize=(12, 6))
    importance.head(15).plot(kind='barh', color='forestgreen', edgecolor='black')
    plt.xlabel('Feature Importance (Gini)', fontweight='bold')
    plt.title('Top 15 Features by Decision Tree Importance', fontweight='bold', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/tree_feature_importance.png", dpi=150)
    plt.close()
    
    return dt_final, optimal_depth, importance

def run_phase_3_enhanced():
    """Main execution function."""
    
    df = load_data()
    
    # PCA Analysis
    pca, X, n_95 = perform_pca_analysis(df)
    
    # Decision Tree Analysis
    dt_model, opt_depth, importance = decision_tree_analysis(df)
    
    print(f"\n{'='*60}")
    print("PCA & Decision Tree Analysis task COMPLETE")
    print(f"{'='*60}")
    
    return pca, dt_model

if __name__ == "__main__":
    run_phase_3_enhanced()
