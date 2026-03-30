# Predicting Workout Calories Burned - ML Analysis

## Project Overview

This project uses machine learning to predict calories burned during workouts. We explored various regression and classification techniques to understand what factors most influence energy expenditure during exercise.

**Authors:**
- Gursahib Singh (TRU ID: T00704197)
- Pooja Verma

**Course:** COMP 4980 - Special Topics: Machine Learning  
**Date:** December 2024

## Dataset

- **Source:** [Kaggle - Predicting Workout Calories Burned](https://www.kaggle.com/code/shreyashpatil217/predicting-workout-calories-burned-xgboost)
- **Size:** 20,000 records × 54 features
- **File:** `dataset.csv` (approximately 10 MB)

## Project Structure

```
project/
├── 01_data_quality.py              # Data loading, cleaning, feature engineering
├── 02_exploratory_analysis.py      # Statistical analysis and visualizations
├── 03_pca_decision_tree.py         # PCA and decision tree exploration
├── 04_ensemble_methods.py          # Random Forest, Gradient Boosting, etc.
├── 05_regularization_svm_nn.py     # Lasso, Ridge, SVM, Neural Networks
├── 06_classification_clustering.py # Classification models and K-Means
├── 07_cross_validation.py          # Cross-validation and final report
├── Report.md                       # Final project report
└── charts/                         # Generated outputs
    ├── *.png                       # Visualizations (charts, plots, diagrams)
    └── text_equivalents/           # Text summaries and accessibility data
        ├── *.md                    # Markdown reports
        └── *.txt                   # Numerical summaries
```

## Environment Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. Clone or download this project folder

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Mac/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Running the Analysis

Run each script in order:

```bash
# Step 1: Data Quality and Feature Engineering
python3 project/01_data_quality.py

# Step 2: Exploratory Data Analysis
python3 project/02_exploratory_analysis.py

# Step 3: PCA and Decision Tree Analysis
python3 project/03_pca_decision_tree.py

# Step 4: Ensemble Methods (Random Forest, Gradient Boosting, etc.)
python3 project/04_ensemble_methods.py

# Step 5: Regularization, SVM, and Neural Networks
python3 project/05_regularization_svm_nn.py

# Step 6: Classification and Clustering
python3 project/06_classification_clustering.py

# Step 7: Cross-Validation Summary
python3 project/07_cross_validation.py
```

### Output Generation

Each Python script generates **dual outputs** for comprehensive documentation and accessibility:

1. **Visual Charts (PNG/SVG)** - Saved to `project/charts/`
   - High-quality visualizations for presentations and reports
   - Includes correlation heatmaps, decision trees, PCA plots, model comparisons, and clustering diagrams
   
2. **Text Equivalents (Markdown/TXT)** - Saved to `project/charts/text_equivalents/`
   - Detailed numerical summaries and interpretations
   - Statistical analysis in text format for accessibility
   - Key findings and insights documented alongside visuals
   - Useful for report generation, accessibility requirements, and archival

This dual-output approach ensures:
- **Accessibility**: Charts are accompanied by text descriptions and data tables
- **Documentation**: All findings are recorded in human-readable format
- **Reproducibility**: Text summaries capture exact numerical results
- **Flexibility**: Different stakeholders can use charts or text based on their needs

## Key Results

- **Best Regression Model:** Gradient Boosting (R² = 0.997)
- **Best Classification Model:** Gradient Boosting (100% accuracy)
- **Key Finding:** Session duration is the most important predictor of calories burned

## Visualizations & Documentation

All analysis outputs are saved in the `project/charts/` directory with a dual-format approach:

### Visual Charts (`project/charts/*.png`)
- Correlation heatmaps and relationship diagrams
- PCA scree plots and biplots
- Decision tree diagrams and tree depth analysis
- Model comparison and performance charts
- Clustering analysis plots and silhouette diagrams
- Feature importance rankings
- Model evaluation metrics visualizations

### Text Equivalents (`project/charts/text_equivalents/`)
- Statistical summaries in Markdown format
- Numerical data and analytical findings in text format
- Data quality reports with detailed metrics
- Model performance tables and comparisons
- Feature correlation lists and interpretations
- Cross-validation results and conclusions

