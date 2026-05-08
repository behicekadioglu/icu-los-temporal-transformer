# icu-los-temporal-transformer
This repository focuses on predicting the Length of Stay (LoS) for patients in the Intensive Care Unit (ICU) using clinical time-series data from the first 24 hours of admission. The project utilizes the MIMIC-IV dataset and implements various baseline models, including Ridge Regression, MLP, and XGBoost, all optimized via Optuna.

## Project Overview
Predicting LoS is a critical task for hospital resource management and patient care planning. Our approach involves:  

- Task: Regression (predicting stay duration in days).  
- Input: 677 clinical parameters (vitals and labs) across a 24-hour window.  
- Optimization: Nested Cross-Validation and Bayesian optimization (Optuna) to prevent data leakage.

## Repository Structure

icu-los-temporal-transformer/
├── data/                   
│   ├── raw/                # Original MIMIC-IV CSV files
│   └── processed/          # Processed 3D and 2D tensors (NumPy format)
├── scripts/                
│   ├── preprocess.py       # Data cleaning, time-alignment, and imputation
│   └── train_baselines.py  # Optimized baseline training loop with Optuna
├── results/                
│   ├── baseline_results/   # Metric outputs in CSV format
│   └── figures/            # Feature importance and performance plots
├── docs/                   
│   └── progress_report.pdf # Project documentation
├── requirements.txt        # Project dependencies
├── .gitignore              # Files to be ignored by Git (e.g., raw data)
└── README.md               # Project documentation and setup guide

## Installation & Setup

1. Clone the Repository:

git clone https://github.com/behicekadioglu/icu-los-temporal-transformer.git
cd icu-los-temporal-transformer

2. Install Dependencies:

It is recommended to use a virtual environment.

pip install -r requirements.txt

3. Data Preparation:
Place the MIMIC-IV clinical database files in data/raw/. Ensure you have access to the dataset through PhysioNet.

## Running Experiments

1. Data Preprocessing
Run the following script to align clinical events into hourly bins and handle missing values:

python scripts/preprocess.py

This will generate .npy files in the data/processed/ directory.  

2. Baseline Training and Optimization
Train the baseline models using Optuna for hyperparameter tuning. You can specify the model and number of trials:

python scripts/train_baselines.py --model all --trials 20 --top_k 30

- --model: Choose from xgboost, mlp, ridge, or all.  
- --trials: Number of optimization attempts per model.
- --top_k: Number of clinical features to select via XGBoost Importance. 

## Initial Results




## Planned Improvements
- Implementation of a Temporal Transformer architecture to capture long-range dependencies in ICU time-series.  
- Development of Attention Maps for clinical interpretability.

## Reproducibility
This project is designed to be fully reproducible. All scaling, feature selection, and hyperparameter tuning are performed strictly within training folds to prevent data leakage. The requirements.txt file ensures environment consistency.

Author: Behice Kadıoğlu
Instructor: Dr. Aytuğ Onan
Institution: İzmir Institute of Technology (İYTE)