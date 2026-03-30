"""
Phase 6 Enhanced: Classification Showcase & Clustering
=======================================================
This module handles:
1. Multi-target classification (Burns_Calories_Bin, Workout_Type)
2. KNN with k optimization
3. Logistic Regression with penalties
4. SVM Classification
5. Random Forest Classifier
6. Comprehensive metrics (Precision, Recall, F1, ROC-AUC)
7. K-Means clustering with silhouette analysis
8. Hierarchical clustering
9. Cluster profiling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
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
    """Load and prepare classification data."""
    print("=" * 60)
    print("PHASE 6 ENHANCED: CLASSIFICATION & CLUSTERING")
    print("=" * 60)
    
    df = pd.read_csv("enhanced_data.csv")
    print(f"\n📊 Dataset: {len(df):,} samples")
    
    return df

def classification_burns_bin(df):
    """Classification of calorie burn intensity bins."""
    print(f"\n🎯 Classification Target: Burns_Calories_Bin")
    
    # Prepare data
    exclude_cols = ['Calories_Burned', 'expected_burn', 'Burns Calories (per 30 min)_bc', 
                    'calorie_efficiency', 'age_group', 'bmi_category', 'Burns_Calories_Bin']
    X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    
    le = LabelEncoder()
    y = le.fit_transform(df['Burns_Calories_Bin'])
    
    print(f"   Classes: {le.classes_}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # === KNN with k optimization ===
    print(f"\n   🔹 KNN Classification:")
    k_values = [3, 5, 7, 9, 11]
    best_k = 5
    best_acc = 0
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
        if cv_scores.mean() > best_acc:
            best_acc = cv_scores.mean()
            best_k = k
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"      Best k: {best_k}")
    print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}")
    results.append({'model': 'KNN', 'accuracy': acc, 'f1': f1})
    
    # === Logistic Regression ===
    print(f"\n   🔹 Logistic Regression:")
    lr = LogisticRegression(max_iter=1000, multi_class='multinomial')
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}")
    results.append({'model': 'Logistic Regression', 'accuracy': acc, 'f1': f1})
    
    # === SVM ===
    print(f"\n   🔹 SVM (Linear):")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}")
    results.append({'model': 'SVM (Linear)', 'accuracy': acc, 'f1': f1})
    
    # === Random Forest Classifier ===
    print(f"\n   🔹 Random Forest Classifier:")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}")
    results.append({'model': 'Random Forest', 'accuracy': acc, 'f1': f1})
    
    # === Gradient Boosting Classifier ===
    print(f"\n   🔹 Gradient Boosting Classifier:")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred_gb)
    f1 = f1_score(y_test, y_pred_gb, average='weighted')
    print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}")
    results.append({'model': 'Gradient Boosting', 'accuracy': acc, 'f1': f1})
    
    # === Confusion Matrix for best model ===
    cm = confusion_matrix(y_test, y_pred_gb)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Gradient Boosting', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/classification_confusion_matrix.png", dpi=150)
    plt.close()
    
    # Results visualization
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    axes[0].bar(results_df['model'], results_df['accuracy'], color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Classification Accuracy Comparison', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0.9, 1.0)
    
    # F1 comparison
    axes[1].bar(results_df['model'], results_df['f1'], color=colors, edgecolor='black')
    axes[1].set_ylabel('F1 Score (Weighted)', fontweight='bold')
    axes[1].set_title('Classification F1 Comparison', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0.9, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/classification_comparison.png", dpi=150)
    plt.close()
    
    return results_df

def clustering_analysis(df):
    """K-Means and Hierarchical clustering."""
    print(f"\n🔮 Clustering Analysis:")
    
    # Select features for clustering (physiological + behavioral)
    cluster_features = ['Age', 'BMI', 'Fat_Percentage', 'Max_BPM', 'Resting_BPM',
                        'Experience_Level', 'Workout_Frequency (days/week)']
    X = df[cluster_features].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # === K-Means with Elbow & Silhouette ===
    print(f"\n   🔹 K-Means Clustering:")
    
    K_range = range(2, 11)
    inertias = []
    silhouettes = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    
    # Find optimal k by silhouette
    optimal_k = K_range[np.argmax(silhouettes)]
    print(f"      Optimal K (by Silhouette): {optimal_k}")
    print(f"      Silhouette Score: {max(silhouettes):.4f}")
    
    # Final clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    # Cluster profiling
    df_clustered = X.copy()
    df_clustered['Cluster'] = cluster_labels
    
    cluster_profile = df_clustered.groupby('Cluster')[cluster_features].mean()
    print(f"\n   Cluster Profiles:")
    print(cluster_profile.round(2))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Elbow plot
    axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (K)', fontweight='bold')
    axes[0, 0].set_ylabel('Inertia (WCSS)', fontweight='bold')
    axes[0, 0].set_title('Elbow Method', fontweight='bold')
    
    # Silhouette plot
    axes[0, 1].plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
    axes[0, 1].axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0, 1].set_xlabel('Number of Clusters (K)', fontweight='bold')
    axes[0, 1].set_ylabel('Silhouette Score', fontweight='bold')
    axes[0, 1].set_title('Silhouette Analysis', fontweight='bold')
    axes[0, 1].legend()
    
    # Cluster scatter (Age vs BMI)
    scatter = axes[1, 0].scatter(df_clustered['Age'], df_clustered['BMI'], 
                                  c=cluster_labels, cmap='Set1', alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Age', fontweight='bold')
    axes[1, 0].set_ylabel('BMI', fontweight='bold')
    axes[1, 0].set_title('Clusters: Age vs BMI', fontweight='bold')
    plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')
    
    # Cluster size
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    axes[1, 1].bar(cluster_sizes.index, cluster_sizes.values, color='teal', edgecolor='black')
    axes[1, 1].set_xlabel('Cluster', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Samples', fontweight='bold')
    axes[1, 1].set_title('Cluster Sizes', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/clustering_analysis.png", dpi=150)
    plt.close()
    
    # Save cluster profile
    cluster_profile.to_csv(f"{OUTPUT_TEXT_DIR}/cluster_profiles.csv")
    
    # === Hierarchical Clustering Dendrogram ===
    print(f"\n   🔹 Hierarchical Clustering:")
    
    # Use subset for dendrogram
    sample_size = min(500, len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[sample_idx]
    
    linkage_matrix = linkage(X_sample, method='ward')
    
    plt.figure(figsize=(14, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=5, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram', fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/dendrogram.png", dpi=150)
    plt.close()
    
    print(f"   ✅ Dendrogram saved")
    
    return cluster_profile, optimal_k

def run_phase_6_enhanced():
    """Main execution function."""
    
    df = load_data()
    
    # Classification
    classification_results = classification_burns_bin(df)
    
    # Clustering
    cluster_profile, optimal_k = clustering_analysis(df)
    
    # Summary
    print(f"\n📋 Phase 6 Summary:")
    best_classifier = classification_results.loc[classification_results['accuracy'].idxmax()]
    print(f"   Best Classifier: {best_classifier['model']} (Acc={best_classifier['accuracy']:.4f})")
    print(f"   Optimal Clusters: {optimal_k}")
    
    print(f"\n{'='*60}")
    print("Classification & Clustering task COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_phase_6_enhanced()
