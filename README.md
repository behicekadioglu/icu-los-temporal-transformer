# icu-los-temporal-transformer
This repository focuses on predicting the Length of Stay (LoS) for patients in the Intensive Care Unit (ICU) using clinical time-series data from the first 24 hours of admission. The project utilizes the MIMIC-IV dataset and implements a custom Temporal Transformer architecture alongside various baseline models (Ridge Regression, MLP, and XGBoost), all optimized via Optuna.


## Project Overview
Predicting LoS is a critical task for hospital resource management and patient care planning. Our approach involves:

- Task: Regression (predicting stay duration in days).
- Input: Top 15 globally selected clinical parameters (vitals and labs) via XGBoost importance, alongside static demographics (Age, Gender) across a 24-hour window.
- Architecture: A PyTorch-based Temporal Transformer with positional encoding, compared against traditional machine learning baselines.
- Optimization: Nested 10-Fold Cross-Validation and Bayesian optimization (Optuna) to prevent data leakage and ensure robust evaluation.


## Repository Structure

```text
icu-los-temporal-transformer/
├── data/                   
│   ├── raw/                # Original MIMIC-IV CSV files (Excluded from Git)
│   └── processed/          # Processed 3D/2D tensors and feature metadata (.npy)
├── scripts/                
│   ├── preprocess.py                  # Data cleaning, global feature selection, and imputation
│   ├── train_baselines.py             # Optimized baseline training loop with Optuna
│   ├── train_transformer.py           # Temporal Transformer training and hyperparameter tuning
│   ├── train_transformer_ablations.py # Ablation studies (Static-only vs Temporal-only)
│   └── visualize_results.py           # Evaluation plots, residual distributions, and feature importance
├── results/                
│   ├── baseline_results/   # Baseline metrics in CSV format
│   ├── transformer_results/# Transformer and ablation metrics in CSV format
│   ├── models/             # Saved PyTorch model weights (.pth) and configs (.json)
│   └── figures/            # Generated plots (Calibration, MAE summary, distributions)
├── docs/                   
│   └── progress_report.pdf # Project documentation and technical reports
├── requirements.txt        
├── .gitignore              
└── README.md               
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



## Running the Pipeline
The project is designed to be executed in a sequential pipeline. Run the following commands from the root directory:


1. Data Preprocessing & Global Feature Selection: Aligns clinical events into hourly bins, handles missing values, and globally selects the top 15 most important features to standardize training across all models. Run the following script to align clinical events into hourly bins and handle missing values:

```bash
python scripts/preprocess.py
```

This will generate .npy files in the data/processed/ directory.  


2. Baseline Training and Optimization: Trains Ridge, MLP, and XGBoost models using Optuna.

```bash
python scripts/train_baselines.py --model all --trials 20
```

- --model: Choose from xgboost, mlp, ridge, or all.  
- --trials: Number of optimization attempts per model.


3. Temporal Transformer Training: Trains the primary deep learning model using the selected temporal and static features.

```bash
python scripts/train_transformer.py
```


4. Ablation Studies: Isolates the impact of data modalities by training constrained versions of the Transformer (static_only and temporal_only).

```bash
python scripts/train_transformer_ablations.py
```


5. Visualization & Evaluation: Generates target distribution, feature importance, regression calibration, and residual plots.

```bash
python scripts/visualize_results.py
```



## Results & Evaluation

Our models are evaluated using a 10-Fold Nested Cross-Validation strategy to ensure the generalizability of our findings. Hyperparameters are automatically tuned via Optuna to maintain a leakage-free and robust evaluation pipeline.


### Baseline Performance Overview
The following table summarizes the initial performance of our optimized baseline models on the MIMIC-IV cohort:

| Model | MAE (Days) | RMSE | R2 Score | MedAE | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **2.40** | 3.57 | -0.73 | 1.50 | Optimized |
| **Ridge** | 2.60 | 3.58 | -0.74 | 1.72 | Optimized |
| **MLP** | 3.06 | 4.28 | -1.75 | 1.93 | Optimized |

(Note: Transformer and Ablation metrics are dynamically saved to results/transformer_results/ after running the respective scripts).




## Reproducibility
This project is designed to be fully reproducible. All scaling and hyperparameter tuning are performed strictly within training folds to prevent data leakage. Global feature selection is finalized prior to cross-validation to establish a standardized input space for all architectures. The requirements.txt file ensures environment consistency.



- Author: Behice Kadıoğlu
- Instructor: Dr. Aytuğ Onan
- Institution: İzmir Institute of Technology (IZTECH)