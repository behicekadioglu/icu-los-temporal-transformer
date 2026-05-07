import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error


SEED = 22

def train_baselines(data_path):
    X = np.load(os.path.join(data_path, 'X_baseline.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    
    # Notebook'undaki modeller ve parametreler
    models = [
        (Ridge(alpha=1.0), "Ridge Regression"),
        (xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED), "XGBoost"),
        (MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED), "Simple MLP")
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    summary_results = []

    for model_obj, name in models:
        print(f"\nTraining {name}...")
        
        fold_metrics = []
        train_times = []
        inf_times = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Calculate Training Time
            start_train = time.time()
            model_obj.fit(X_train, y_train)
            train_times.append(time.time() - start_train)

            # Calculate Inference Time Per Sample
            start_inf = time.time()
            preds = model_obj.predict(X_val)
            inf_times.append((time.time() - start_inf) / len(y_val))

            fold_metrics.append({
                'MAE': mean_absolute_error(y_val, preds),
                'RMSE': np.sqrt(mean_squared_error(y_val, preds)),
                'R2': r2_score(y_val, preds),
                'MedAE': median_absolute_error(y_val, preds)
            })

        # Calculate Average Metrics Across Folds
        avg_res = pd.DataFrame(fold_metrics).mean().to_dict()
        avg_res['Model'] = name
        avg_res['Avg_Train_Time_Sec'] = np.mean(train_times)
        avg_res['Avg_Inf_Time_Per_Sample'] = np.mean(inf_times)
        summary_results.append(avg_res)

    # Save Results
    df_final = pd.DataFrame(summary_results)
    df_final = df_final[['Model', 'MAE', 'RMSE', 'R2', 'MedAE', 'Avg_Train_Time_Sec', 'Avg_Inf_Time_Per_Sample']]
    
    df_final.to_csv('baseline_results_summary.csv', index=False)
    
    print("\n" + "="*50)
    print("FINAL BASELINE COMPARISON TABLE")
    print("="*50)
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    train_baselines("data/processed/")