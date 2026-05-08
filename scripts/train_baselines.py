import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
import optuna
import argparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error

SEED = 22

def select_features_in_fold(X_train_flat, y_train, feature_names, top_k):
    """
    Identifies the most important features using XGBoost within the training fold only.
    Aggregates importance scores across all 24 time steps.
    """
    fs_model = xgb.XGBRegressor(n_estimators=50, random_state=SEED)
    fs_model.fit(X_train_flat, y_train)
    
    importances = fs_model.feature_importances_
    num_features = len(feature_names)
    item_importances = {}
    
    for i, name in enumerate(feature_names):
        # sum importance of a single feature across its 24 hourly occurrences
        total_imp = sum(importances[i::num_features])
        item_importances[name] = total_imp
        
    sorted_items = sorted(item_importances.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:top_k]]

def objective(trial, X_train, y_train, model_type):
    """
    Optuna objective function to minimize Mean Absolute Error (MAE).
    """
    # Split training fold further into inner train/val for tuning (Nested CV approach)
    x_t, x_v, y_t, y_v = train_test_split(X_train, y_train, test_test_size=0.2, random_state=SEED)

    if model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': SEED
        }
        model = xgb.XGBRegressor(**params)
        
    elif model_type == 'ridge':
        params = {
            'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True)
        }
        model = Ridge(**params)
        
    elif model_type == 'mlp':
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64,), (128, 64), (256, 128, 64)]),
            'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),
            'max_iter': 500,
            'random_state': SEED
        }
        model = MLPRegressor(**params)

    model.fit(x_t, y_t)
    preds = model.predict(x_v)
    return mean_absolute_error(y_v, preds)

def train_with_optuna(data_path, selected_model, top_k=30, n_trials=20):
    # Load the raw tensors prepared by preprocess.py
    X_temp_raw = np.load(os.path.join(data_path, 'X_temporal_raw.npy'))
    X_static_raw = np.load(os.path.join(data_path, 'X_static_raw.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    f_names = np.load(os.path.join(data_path, 'feature_names.npy'))
    
    models_list = ['ridge', 'xgboost', 'mlp'] if selected_model == 'all' else [selected_model]
    
    # 10-Fold Cross Validation loop
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    final_summary = []

    for m_type in models_list:
        print(f"\n--- Starting Optimization for Model: {m_type.upper()} ---")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp_raw)):
            # 1. Leakage-Free Scaling
            scaler = MinMaxScaler()
            N_tr, T, F = X_temp_raw[train_idx].shape
            X_tr_scaled = scaler.fit_transform(X_temp_raw[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
            
            N_val = len(val_idx)
            X_val_scaled = scaler.transform(X_temp_raw[val_idx].reshape(-1, F)).reshape(N_val, T, F)
            
            # 2. Leakage-Free Feature Selection
            X_tr_flat = X_tr_scaled.reshape(N_tr, -1)
            selected_items = select_features_in_fold(X_tr_flat, y[train_idx], f_names, top_k)
            
            sel_idx = [i for i, name in enumerate(f_names) if name in selected_items]
            X_tr_final = X_tr_scaled[:, :, sel_idx].reshape(N_tr, -1)
            X_val_final = X_val_scaled[:, :, sel_idx].reshape(N_val, -1)
            
            # Add static demographics
            X_tr_final = np.hstack([X_tr_final, X_static_raw[train_idx]])
            X_val_final = np.hstack([X_val_final, X_static_raw[val_idx]])
            
            # 3. Optuna Hyperparameter Optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X_tr_final, y[train_idx], m_type), n_trials=n_trials)
            
            # 4. Train final model with best parameters
            best_params = study.best_params
            if m_type == 'xgboost':
                final_model = xgb.XGBRegressor(**best_params, random_state=SEED)
            elif m_type == 'ridge':
                final_model = Ridge(**best_params)
            elif m_type == 'mlp':
                final_model = MLPRegressor(**best_params, random_state=SEED)

            start_train = time.time()
            final_model.fit(X_tr_final, y[train_idx])
            t_time = time.time() - start_train
            
            # 5. Inference
            start_inf = time.time()
            preds = final_model.predict(X_val_final)
            inf_time = (time.time() - start_inf) / len(val_idx)
            
            fold_metrics.append({
                'MAE': mean_absolute_error(y[val_idx], preds),
                'RMSE': np.sqrt(mean_squared_error(y[val_idx], preds)),
                'R2': r2_score(y[val_idx], preds),
                'MedAE': median_absolute_error(y[val_idx], preds),
                'Train_Time': t_time,
                'Inf_Time': inf_time
            })
            print(f"Fold {fold+1} Optimized (MAE: {fold_metrics[-1]['MAE']:.4f})")

        avg_res = pd.DataFrame(fold_metrics).mean().to_dict()
        avg_res['Model'] = m_type
        final_summary.append(avg_res)

    # Save results to the structured directory
    output_folder = 'results/baseline_results'
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    df_results = pd.DataFrame(final_summary)
    df_results.to_csv(os.path.join(output_folder, f'optimized_results_{selected_model}.csv'), index=False)
    
    print("\n" + "="*60 + "\nFINAL OPTIMIZED RESULTS\n" + "="*60)
    print(df_results[['Model', 'MAE', 'RMSE', 'R2', 'MedAE']].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', choices=['xgboost', 'mlp', 'ridge', 'all'])
    parser.add_argument('--trials', type=int, default=20, help="Number of Optuna trials per model")
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--data_path', type=str, default='data/processed/')

    args = parser.parse_args()
    train_with_optuna(args.data_path, args.model, args.top_k, args.trials)