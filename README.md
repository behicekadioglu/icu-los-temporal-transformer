# icu-los-temporal-transformer
This repository focuses on predicting the Length of Stay (LoS) for patients in the Intensive Care Unit (ICU) using clinical time-series data from the first 24 hours of admission. The project utilizes the MIMIC-IV dataset and implements various baseline models, including Ridge Regression, MLP, and XGBoost, all optimized via Optuna.

## Project Overview
Predicting LoS is a critical task for hospital resource management and patient care planning. Our approach involves:  

- Task: Regression (predicting stay duration in days).  
- Input: 677 clinical parameters (vitals and labs) across a 24-hour window.  
- Optimization: Nested Cross-Validation and Bayesian optimization (Optuna) to prevent data leakage.

## Repository Structure

```text
icu-los-temporal-transformer/
├── data/                   
│   ├── raw/                # Original MIMIC-IV CSV files (Excluded from Git)
│   └── processed/          # Processed 3D and 2D tensors in NumPy format
├── scripts/                
│   ├── preprocess.py       # Data cleaning, time-alignment, and imputation
│   └── train_baselines.py  # Optimized baseline training loop with Optuna
├── results/                
│   ├── baseline_results/   # Metric outputs (MAE, RMSE, R2) in CSV format
│   └── figures/            # Feature importance and performance visualizations
├── docs/                   
│   └── progress_report.pdf # Project documentation and technical reports
├── requirements.txt        # Project dependencies and library versions
├── .gitignore              # Files to be ignored by Git (e.g., raw data, .npy files)
└── README.md               # Project documentation and setup guide
```

## Installation & Setup

1. Clone the Repository:


```bash
git clone https://github.com/behicekadioglu/icu-los-temporal-transformer.git
cd icu-los-temporal-transformer
```


2. Install Dependencies:

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

3. Data Preparation:
Place the MIMIC-IV clinical database files in data/raw/. Ensure you have access to the dataset through PhysioNet.

## Running Experiments

1. Data Preprocessing
Run the following script to align clinical events into hourly bins and handle missing values:

```bash
python scripts/preprocess.py
```

This will generate .npy files in the data/processed/ directory.  

2. Baseline Training and Optimization
Train the baseline models using Optuna for hyperparameter tuning. You can specify the model and number of trials:

```bash
python scripts/train_baselines.py --model all --trials 20 --top_k 30
```

- --model: Choose from xgboost, mlp, ridge, or all.  
- --trials: Number of optimization attempts per model.
- --top_k: Number of clinical features to select via XGBoost Importance. 

## Initial Results

Our baseline models were evaluated using a 10-Fold Nested Cross-Validation strategy to ensure the generalizability of our findings. Hyperparameters were automatically tuned via Optuna to maintain a leakage-free and robust evaluation pipeline.

### Optimized Performance Metrics
The following table summarizes the performance of our optimized baseline models on the MIMIC-IV cohort:

| Model | MAE (Days) | RMSE | R2 Score | MedAE | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **2.40** | 3.57 | -0.73 | 1.50 | Optimized |
| **Ridge** | 2.60 | 3.58 | -0.74 | 1.72 | Optimized |
| **MLP** | 3.06 | 4.28 | -1.75 | 1.93 | Optimized |

### Interpretation of Results
* Best Performance: XGBoost is currently our strongest baseline, achieving the lowest error with an MAE of **2.40 days**. This suggests that gradient-boosted decision trees are better suited for capturing non-linear signals in flattened clinical data compared to the linear Ridge model or the feed-forward MLP.
* Error Distribution: The discrepancy between MAE and MedAE (e.g., 2.40 vs 1.50 for XGBoost) points to the presence of clinical outliers—patients with exceptionally long ICU stays that significantly inflate the mean error.
* Motivation for Transformer: The negative $R^2$ scores across all static models highlight the inherent difficulty of predicting ICU stay duration. This indicates that static baseline models struggle to outperform a mean-based prediction in terms of explained variance, likely due to the high volatility of patient health trajectories. This underscores the technical need for the next phase of the project: the Temporal Transformer, which is specifically designed to handle such complex temporal dependencies.



## Planned Improvements
- Implementation of a Temporal Transformer architecture to capture long-range dependencies in ICU time-series.  
- Development of Attention Maps for clinical interpretability.

## Reproducibility
This project is designed to be fully reproducible. All scaling, feature selection, and hyperparameter tuning are performed strictly within training folds to prevent data leakage. The requirements.txt file ensures environment consistency.



Author: Behice Kadıoğlu
Instructor: Dr. Aytuğ Onan
Institution: İzmir Institute of Technology (IZTECH)