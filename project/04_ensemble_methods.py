"""
Phase 4 Enhanced: Advanced Ensemble Methods & Hyperparameter Tuning
====================================================================
This module handles:
1. Random Forest with feature importance
2. ExtraTrees comparison
3. AdaBoost Regressor
4. Gradient Boosting with learning curve
5. Stacking & Voting Ensembles
6. GridSearchCV hyperparameter optimization
7. Comprehensive model comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                               GradientBoostingRegressor, AdaBoostRegressor,
                               VotingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
    print("PHASE 4 ENHANCED: ADVANCED ENSEMBLE METHODS")
    print("=" * 60)
    
    df = pd.read_csv("enhanced_data.csv")
    
    # Prepare features (exclude target and leaky features)
    exclude_cols = ['Calories_Burned', 'expected_burn', 'Burns Calories (per 30 min)_bc', 
                    'calorie_efficiency', 'age_group', 'bmi_category']
    X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    y = df['Calories_Burned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Also create scaled version
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Dataset prepared:")
    print(f"   Training: {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"   Testing: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X.columns

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train and analyze Random Forest."""
    print(f"\n🌲 Random Forest Regressor:")
    
    # Train with good hyperparameters
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, max_features='sqrt',
                                min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    print(f"   Test MAE: {test_mae:.2f}")
    
    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\n   Top 5 Features:")
    for feat in importance.head(5).index:
        print(f"      {feat}: {importance[feat]:.4f}")
    
    return rf, {'model': 'Random Forest', 'train_r2': train_r2, 'test_r2': test_r2, 
                'rmse': test_rmse, 'mae': test_mae}, importance

def train_extra_trees(X_train, X_test, y_train, y_test, feature_names):
    """Train ExtraTrees (Extremely Randomized Trees)."""
    print(f"\n🌳 ExtraTrees Regressor:")
    
    et = ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    et.fit(X_train, y_train)
    
    y_pred_test = et.predict(X_test)
    
    train_r2 = et.score(X_train, y_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    
    return et, {'model': 'ExtraTrees', 'train_r2': train_r2, 'test_r2': test_r2, 
                'rmse': test_rmse, 'mae': test_mae}

def train_adaboost(X_train, X_test, y_train, y_test):
    """Train AdaBoost Regressor."""
    print(f"\n🚀 AdaBoost Regressor:")
    
    ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    ada.fit(X_train, y_train)
    
    y_pred_test = ada.predict(X_test)
    
    train_r2 = ada.score(X_train, y_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    
    return ada, {'model': 'AdaBoost', 'train_r2': train_r2, 'test_r2': test_r2, 
                 'rmse': test_rmse, 'mae': test_mae}

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting with learning curve."""
    print(f"\n📈 Gradient Boosting Regressor:")
    
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1,
                                    min_samples_split=5, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred_test = gb.predict(X_test)
    
    train_r2 = gb.score(X_train, y_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    
    # Staged predictions for learning curve
    print(f"\n   Generating Learning Curve...")
    train_scores = []
    test_scores = []
    for i, y_pred_staged in enumerate(gb.staged_predict(X_test)):
        if (i + 1) % 10 == 0:
            test_scores.append(r2_score(y_test, y_pred_staged))
    
    return gb, {'model': 'Gradient Boosting', 'train_r2': train_r2, 'test_r2': test_r2, 
                'rmse': test_rmse, 'mae': test_mae}, test_scores

def train_stacking_ensemble(X_train, X_test, y_train, y_test):
    """Train Stacking Ensemble."""
    print(f"\n📚 Stacking Regressor:")
    
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42))
    ]
    
    stacking = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0))
    stacking.fit(X_train, y_train)
    
    y_pred_test = stacking.predict(X_test)
    
    train_r2 = stacking.score(X_train, y_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    
    return stacking, {'model': 'Stacking', 'train_r2': train_r2, 'test_r2': test_r2, 
                      'rmse': test_rmse, 'mae': test_mae}

def train_voting_ensemble(X_train, X_test, y_train, y_test):
    """Train Voting Ensemble."""
    print(f"\n🗳️ Voting Regressor:")
    
    voting = VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42))
    ])
    voting.fit(X_train, y_train)
    
    y_pred_test = voting.predict(X_test)
    
    train_r2 = voting.score(X_train, y_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    
    return voting, {'model': 'Voting', 'train_r2': train_r2, 'test_r2': test_r2, 
                    'rmse': test_rmse, 'mae': test_mae}

def hyperparameter_tuning(X_train, y_train):
    """GridSearchCV for Gradient Boosting."""
    print(f"\n🔧 Hyperparameter Tuning (GridSearchCV):")
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"   Best Parameters: {grid_search.best_params_}")
    print(f"   Best CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def create_visualizations(results_df, rf_importance, feature_names):
    """Create comprehensive visualizations."""
    print(f"\n📊 Creating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Model Comparison - R² Score
    models = results_df['model'].values
    test_r2 = results_df['test_r2'].values
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    bars = axes[0, 0].bar(models, test_r2, color=colors, edgecolor='black')
    axes[0, 0].set_title('Model Comparison: Test R² Score', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_ylim(0.9, 1.0)
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar, score in zip(bars, test_r2):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{score:.4f}', ha='center', fontweight='bold', fontsize=9)
    
    # 2. RMSE Comparison
    rmse = results_df['rmse'].values
    axes[0, 1].bar(models, rmse, color='coral', edgecolor='black')
    axes[0, 1].set_title('Model Comparison: RMSE (Lower is Better)', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Train vs Test R² (Overfitting Check)
    train_r2 = results_df['train_r2'].values
    x_pos = np.arange(len(models))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, train_r2, width, label='Train R²', color='steelblue', edgecolor='black')
    axes[1, 0].bar(x_pos + width/2, test_r2, width, label='Test R²', color='darkorange', edgecolor='black')
    axes[1, 0].set_title('Train vs Test R² (Overfitting Check)', fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].legend()
    
    # 4. Feature Importance (RF)
    rf_importance.head(10).plot(kind='barh', ax=axes[1, 1], color='forestgreen', edgecolor='black')
    axes[1, 1].set_title('Top 10 Features (Random Forest)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/ensemble_comparison.png", dpi=150)
    plt.close()
    
    print(f"   ✅ Visualizations saved")

def run_phase_4_enhanced():
    """Main execution function."""
    
    # Load data
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, feature_names = load_data()
    
    results = []
    
    # Train models
    rf, rf_results, rf_importance = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    results.append(rf_results)
    
    et, et_results = train_extra_trees(X_train, X_test, y_train, y_test, feature_names)
    results.append(et_results)
    
    ada, ada_results = train_adaboost(X_train, X_test, y_train, y_test)
    results.append(ada_results)
    
    gb, gb_results, gb_curve = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results.append(gb_results)
    
    stacking, stacking_results = train_stacking_ensemble(X_train, X_test, y_train, y_test)
    results.append(stacking_results)
    
    voting, voting_results = train_voting_ensemble(X_train, X_test, y_train, y_test)
    results.append(voting_results)
    
    # Hyperparameter tuning
    best_gb, best_params = hyperparameter_tuning(X_train, y_train)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print(f"\n📋 Model Comparison Summary:")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(f"{OUTPUT_TEXT_DIR}/ensemble_results.csv", index=False)
    
    # Create visualizations
    create_visualizations(results_df, rf_importance, feature_names)
    
    print(f"\n{'='*60}")
    print("Ensemble Methods Analysis task COMPLETE")
    print(f"{'='*60}")
    
    return results_df

if __name__ == "__main__":
    run_phase_4_enhanced()
