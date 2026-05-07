import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
import argparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error

def train_baselines(data_path, selected_model):
    X = np.load(os.path.join(data_path, 'X_baseline.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    
    # Description of available models
    all_models = {
        'ridge': (Ridge(alpha=1.0), "Ridge Regression"),
        'xgboost': (xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42), "XGBoost"),
        'mlp': (MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42), "Simple MLP")
    }

    # Description of selected model
    if selected_model == 'all':
        models_to_train = list(all_models.values())
    else:
        models_to_train = [all_models[selected_model]]


    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    summary_results = []

    for model_obj, name in models_to_train:
        print(f"\nTraining {name}...")
        
        fold_metrics = []
        train_times = []
        inf_times = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Training Times
            start_train = time.time()
            model_obj.fit(X_train, y_train)
            train_times.append(time.time() - start_train)

            # Inference Times
            start_inf = time.time()
            preds = model_obj.predict(X_val)
            inf_times.append((time.time() - start_inf) / len(y_val))

            fold_metrics.append({
                'MAE': mean_absolute_error(y_val, preds),
                'RMSE': np.sqrt(mean_squared_error(y_val, preds)),
                'R2': r2_score(y_val, preds),
                'MedAE': median_absolute_error(y_val, preds)
            })
            print(f"  Fold {fold+1} completed.")

        # Averaging Metrics
        avg_res = pd.DataFrame(fold_metrics).mean().to_dict()
        avg_res['Model'] = name
        avg_res['Avg_Train_Time_Sec'] = np.mean(train_times)
        avg_res['Avg_Inf_Time_Per_Sample'] = np.mean(inf_times)
        summary_results.append(avg_res)

    # Save and Print Results
    df_final = pd.DataFrame(summary_results)
    df_final = df_final[['Model', 'MAE', 'RMSE', 'R2', 'MedAE', 'Avg_Train_Time_Sec', 'Avg_Inf_Time_Per_Sample']]
    
    output_name = f'baseline_results_{selected_model}.csv'
    df_final.to_csv(output_name, index=False)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR: {selected_model.upper()}")
    print("="*50)
    print(df_final.to_string(index=False))
    print(f"\nResults saved to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', 
                        choices=['xgboost', 'mlp', 'ridge', 'all'])
    parser.add_argument('--data_path', type=str, default='data/processed/')
    
    args = parser.parse_args()
    train_baselines(args.data_path, args.model)
