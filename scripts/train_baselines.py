import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
import argparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error


SEED = 22


def select_features_in_fold(X_train_flat, y_train, feature_names, top_k):
    """
    Performs feature selection using XGBoost importance scores 
    on the current training fold only.
    """
    fs_model = xgb.XGBRegressor(n_estimators=100, random_state=SEED)
    fs_model.fit(X_train_flat, y_train)
    
    importances = fs_model.feature_importances_
    num_features = len(feature_names)
    item_importances = {}
    
    for i, name in enumerate(feature_names):
        # Sum importance scores for each feature across all 24 hours
        total_imp = sum(importances[i::num_features])
        item_importances[name] = total_imp
        
    # Sort items by aggregated importance
    sorted_items = sorted(item_importances.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:top_k]]



def train_baselines(data_path, selected_model, top_k=30):
    # Load raw preprocessed data
    X_temp_raw = np.load(os.path.join(data_path, 'X_temporal_raw.npy'))
    X_static_raw = np.load(os.path.join(data_path, 'X_static_raw.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    f_names = np.load(os.path.join(data_path, 'feature_names.npy'))
    

    # Model definitions
    all_models = {
        'ridge': (Ridge(alpha=1.0), "Ridge Regression"),
        'xgboost': (xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED), "XGBoost"),
        'mlp': (MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED), "Simple MLP")
    }


    if selected_model == 'all':
        models_to_train = list(all_models.values())
    else:
        models_to_train = [all_models[selected_model]]


    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    final_summary = []

    for model_obj, model_name in models_to_train:
        print(f"\nEvaluating Model: {model_name} (Top K={top_k})")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp_raw)):
            # 1. Split data
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 2. Scaling (Fit ONLY on training fold to prevent leakage)
            scaler = MinMaxScaler()
            N_tr, T, F = X_temp_raw[train_idx].shape
            X_tr_scaled = scaler.fit_transform(X_temp_raw[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
            
            N_val = len(val_idx)
            X_val_scaled = scaler.transform(X_temp_raw[val_idx].reshape(-1, F)).reshape(N_val, T, F)
            
            # 3. Feature Selection (Fit ONLY on training fold)
            X_tr_flat = X_tr_scaled.reshape(N_tr, -1)
            selected_items = select_features_in_fold(X_tr_flat, y_train, f_names, top_k)
            
            # Filter temporal features based on selection
            sel_idx = [i for i, name in enumerate(f_names) if name in selected_items]
            X_tr_final = X_tr_scaled[:, :, sel_idx].reshape(N_tr, -1)
            X_val_final = X_val_scaled[:, :, sel_idx].reshape(N_val, -1)
            
            # 4. Attach Static Features
            X_tr_final = np.hstack([X_tr_final, X_static_raw[train_idx]])
            X_val_final = np.hstack([X_val_final, X_static_raw[val_idx]])
            
            # 5. Training and Timing
            start_train = time.time()
            model_obj.fit(X_tr_final, y_train)
            train_time = time.time() - start_train
            
            # 6. Inference and Evaluation
            start_inf = time.time()
            preds = model_obj.predict(X_val_final)
            inf_time = (time.time() - start_inf) / len(y_val)
            
            fold_metrics.append({
                'MAE': mean_absolute_error(y_val, preds),
                'RMSE': np.sqrt(mean_squared_error(y_val, preds)),
                'R2': r2_score(y_val, preds),
                'MedAE': median_absolute_error(y_val, preds),
                'Train_Time': train_time,
                'Inf_Time': inf_time
            })
            print(f"  Fold {fold+1} completed.")

        # Aggregate metrics for the model
        avg_res = pd.DataFrame(fold_metrics).mean().to_dict()
        avg_res['Model'] = model_name
        final_summary.append(avg_res)

    # Presentation
    df_results = pd.DataFrame(final_summary)
    df_results = df_results[['Model', 'MAE', 'RMSE', 'R2', 'MedAE', 'Train_Time', 'Inf_Time']]
    
    print("\n" + "="*60)
    print("LEAKAGE-FREE BASELINE RESULTS")
    print("="*60)
    print(df_results.to_string(index=False))
    

    # Önce klasörün var olduğundan emin ol
    output_folder = 'results/baseline_results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dosyayı yeni klasöre kaydet
    df_results.to_csv(os.path.join(output_folder, f'baseline_results_{selected_model}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', choices=['xgboost', 'mlp', 'ridge', 'all'])
    parser.add_argument('--top_k', type=int, default=30, help="Number of items to select")
    parser.add_argument('--data_path', type=str, default='data/processed/')
    
    args = parser.parse_args()
    train_baselines(args.data_path, args.model, args.top_k)