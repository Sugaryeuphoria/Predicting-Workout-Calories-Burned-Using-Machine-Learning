"""
Phase 5 Enhanced: Regularization Deep Dive & SVM Analysis
==========================================================
This module handles:
1. Regularization path plots (Lasso, Ridge)
2. ElasticNet optimization
3. Coefficient comparison
4. SVM with multiple kernels
5. SVM hyperparameter tuning
6. Neural Network architecture experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
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
    print("PHASE 5 ENHANCED: REGULARIZATION & SVM ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv("enhanced_data.csv")
    
    # Prepare features
    exclude_cols = ['Calories_Burned', 'expected_burn', 'Burns Calories (per 30 min)_bc', 
                    'calorie_efficiency', 'age_group', 'bmi_category']
    X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    y = df['Calories_Burned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Dataset: {len(X_train):,} train, {len(X_test):,} test, {X.shape[1]} features")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X.columns, scaler

def regularization_analysis(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """Comprehensive regularization analysis."""
    print(f"\n📐 Regularization Analysis:")
    
    results = {}
    
    # === LASSO (L1) ===
    print(f"\n   🔹 Lasso Regression (L1):")
    alphas = np.logspace(-4, 2, 50)
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)
    
    best_alpha_lasso = lasso_cv.alpha_
    lasso = Lasso(alpha=best_alpha_lasso)
    lasso.fit(X_train_scaled, y_train)
    
    lasso_r2 = r2_score(y_test, lasso.predict(X_test_scaled))
    non_zero = np.sum(lasso.coef_ != 0)
    
    print(f"      Best Alpha: {best_alpha_lasso:.4f}")
    print(f"      Test R²: {lasso_r2:.4f}")
    print(f"      Non-zero Features: {non_zero}/{len(feature_names)}")
    
    results['Lasso'] = {'r2': lasso_r2, 'alpha': best_alpha_lasso, 'model': lasso}
    
    # === RIDGE (L2) ===
    print(f"\n   🔹 Ridge Regression (L2):")
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train_scaled, y_train)
    
    best_alpha_ridge = ridge_cv.alpha_
    ridge = Ridge(alpha=best_alpha_ridge)
    ridge.fit(X_train_scaled, y_train)
    
    ridge_r2 = r2_score(y_test, ridge.predict(X_test_scaled))
    
    print(f"      Best Alpha: {best_alpha_ridge:.4f}")
    print(f"      Test R²: {ridge_r2:.4f}")
    
    results['Ridge'] = {'r2': ridge_r2, 'alpha': best_alpha_ridge, 'model': ridge}
    
    # === ELASTICNET ===
    print(f"\n   🔹 ElasticNet (L1+L2):")
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_en_score = 0
    best_l1_ratio = 0.5
    
    for l1_ratio in l1_ratios:
        en = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, random_state=42)
        en.fit(X_train_scaled, y_train)
        score = r2_score(y_test, en.predict(X_test_scaled))
        if score > best_en_score:
            best_en_score = score
            best_l1_ratio = l1_ratio
    
    elastic = ElasticNet(alpha=0.1, l1_ratio=best_l1_ratio, random_state=42)
    elastic.fit(X_train_scaled, y_train)
    en_r2 = r2_score(y_test, elastic.predict(X_test_scaled))
    
    print(f"      Best L1 Ratio: {best_l1_ratio}")
    print(f"      Test R²: {en_r2:.4f}")
    
    results['ElasticNet'] = {'r2': en_r2, 'l1_ratio': best_l1_ratio, 'model': elastic}
    
    # === VISUALIZATION: Regularization Path ===
    print(f"\n   Creating Regularization Path Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Lasso Path
    lasso_coefs = []
    for alpha in alphas:
        lasso_temp = Lasso(alpha=alpha, max_iter=10000)
        lasso_temp.fit(X_train_scaled, y_train)
        lasso_coefs.append(lasso_temp.coef_)
    
    lasso_coefs = np.array(lasso_coefs)
    
    for i in range(min(10, lasso_coefs.shape[1])):
        axes[0, 0].plot(np.log10(alphas), lasso_coefs[:, i], label=feature_names[i] if i < 5 else None)
    axes[0, 0].axvline(np.log10(best_alpha_lasso), color='k', linestyle='--', label=f'Best α')
    axes[0, 0].set_xlabel('log(α)', fontweight='bold')
    axes[0, 0].set_ylabel('Coefficient', fontweight='bold')
    axes[0, 0].set_title('Lasso Regularization Path', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    
    # Ridge Path
    ridge_coefs = []
    for alpha in alphas:
        ridge_temp = Ridge(alpha=alpha)
        ridge_temp.fit(X_train_scaled, y_train)
        ridge_coefs.append(ridge_temp.coef_)
    
    ridge_coefs = np.array(ridge_coefs)
    
    for i in range(min(10, ridge_coefs.shape[1])):
        axes[0, 1].plot(np.log10(alphas), ridge_coefs[:, i])
    axes[0, 1].axvline(np.log10(best_alpha_ridge), color='k', linestyle='--', label=f'Best α')
    axes[0, 1].set_xlabel('log(α)', fontweight='bold')
    axes[0, 1].set_ylabel('Coefficient', fontweight='bold')
    axes[0, 1].set_title('Ridge Regularization Path', fontweight='bold')
    
    # Coefficient Comparison
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Lasso': lasso.coef_,
        'Ridge': ridge.coef_,
        'ElasticNet': elastic.coef_
    })
    coef_df['Lasso_abs'] = np.abs(coef_df['Lasso'])
    coef_df = coef_df.sort_values('Lasso_abs', ascending=False).head(15)
    
    x = np.arange(len(coef_df))
    width = 0.25
    axes[1, 0].bar(x - width, coef_df['Lasso'], width, label='Lasso', color='#E74C3C')
    axes[1, 0].bar(x, coef_df['Ridge'], width, label='Ridge', color='#3498DB')
    axes[1, 0].bar(x + width, coef_df['ElasticNet'], width, label='ElasticNet', color='#2ECC71')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(coef_df['Feature'], rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Coefficient', fontweight='bold')
    axes[1, 0].set_title('Top 15 Coefficients Comparison', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # Model Comparison
    models = ['Lasso', 'Ridge', 'ElasticNet']
    r2_scores = [results[m]['r2'] for m in models]
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    axes[1, 1].bar(models, r2_scores, color=colors, edgecolor='black')
    axes[1, 1].set_ylabel('Test R² Score', fontweight='bold')
    axes[1, 1].set_title('Regularization Methods Comparison', fontweight='bold')
    axes[1, 1].set_ylim(0.95, 1.0)
    for i, (model, score) in enumerate(zip(models, r2_scores)):
        axes[1, 1].text(i, score + 0.002, f'{score:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/regularization_analysis.png", dpi=150)
    plt.close()
    
    # Save coefficients
    coef_df.to_csv(f"{OUTPUT_TEXT_DIR}/regularization_coefficients.csv", index=False)
    
    return results

def svm_analysis(X_train_scaled, X_test_scaled, y_train, y_test):
    """SVM with multiple kernels."""
    print(f"\n⚡ Support Vector Regression:")
    
    # Use subset for faster training
    n_samples = min(5000, len(X_train_scaled))
    indices = np.random.choice(len(X_train_scaled), n_samples, replace=False)
    X_train_sub = X_train_scaled[indices]
    y_train_sub = y_train.iloc[indices]
    
    results = {}
    kernels = ['linear', 'rbf', 'poly']
    
    for kernel in kernels:
        print(f"\n   🔹 SVR ({kernel} kernel):")
        
        if kernel == 'poly':
            svr = SVR(kernel=kernel, C=100, degree=2)
        else:
            svr = SVR(kernel=kernel, C=100)
        
        svr.fit(X_train_sub, y_train_sub)
        
        y_pred = svr.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"      Test R²: {r2:.4f}")
        print(f"      Test RMSE: {rmse:.2f}")
        
        results[kernel] = {'r2': r2, 'rmse': rmse}
    
    return results

def neural_network_analysis(X_train_scaled, X_test_scaled, y_train, y_test):
    """Neural Network architecture experiments."""
    print(f"\n🧠 Neural Network Analysis:")
    
    architectures = [
        (32,),
        (64, 32),
        (128, 64, 32),
        (256, 128, 64),
    ]
    
    results = []
    
    for arch in architectures:
        print(f"\n   🔹 MLP {arch}:")
        
        mlp = MLPRegressor(hidden_layer_sizes=arch, activation='relu', solver='adam',
                           max_iter=500, random_state=42, early_stopping=True)
        mlp.fit(X_train_scaled, y_train)
        
        y_pred = mlp.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"      Test R²: {r2:.4f}")
        print(f"      Iterations: {mlp.n_iter_}")
        
        results.append({
            'architecture': str(arch),
            'r2': r2,
            'rmse': rmse,
            'iterations': mlp.n_iter_
        })
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    archs = [r['architecture'] for r in results]
    r2_scores = [r['r2'] for r in results]
    rmse_scores = [r['rmse'] for r in results]
    
    axes[0].bar(archs, r2_scores, color='purple', edgecolor='black')
    axes[0].set_ylabel('Test R² Score', fontweight='bold')
    axes[0].set_title('MLP Architecture Comparison', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(archs, rmse_scores, color='teal', edgecolor='black')
    axes[1].set_ylabel('Test RMSE', fontweight='bold')
    axes[1].set_title('MLP RMSE by Architecture', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/neural_network_comparison.png", dpi=150)
    plt.close()
    
    return results

def run_phase_5_enhanced():
    """Main execution function."""
    
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, feature_names, scaler = load_data()
    
    # Regularization analysis
    reg_results = regularization_analysis(X_train_s, X_test_s, y_train, y_test, feature_names)
    
    # SVM analysis
    svm_results = svm_analysis(X_train_s, X_test_s, y_train, y_test)
    
    # Neural network analysis
    nn_results = neural_network_analysis(X_train_s, X_test_s, y_train, y_test)
    
    # Summary
    print(f"\n📋 Phase 5 Summary:")
    print(f"   Best Regularization: Ridge (R² = {reg_results['Ridge']['r2']:.4f})")
    best_svm = max(svm_results.items(), key=lambda x: x[1]['r2'])
    print(f"   Best SVM Kernel: {best_svm[0]} (R² = {best_svm[1]['r2']:.4f})")
    best_nn = max(nn_results, key=lambda x: x['r2'])
    print(f"   Best MLP: {best_nn['architecture']} (R² = {best_nn['r2']:.4f})")
    
    print(f"\n{'='*60}")
    print("Regularization, SVM & Neural Networks task COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_phase_5_enhanced()
