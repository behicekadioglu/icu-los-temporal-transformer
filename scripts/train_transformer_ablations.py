import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from train_transformer import TemporalTransformer, MIMICDataset, evaluate_all_metrics

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

def select_features_in_fold(X_train_flat, y_train, feature_names, top_k):
    fs_model = xgb.XGBRegressor(n_estimators=50, random_state=SEED)
    fs_model.fit(X_train_flat, y_train)
    importances, num_f = fs_model.feature_importances_, len(feature_names)
    item_imp = {name: sum(importances[i::num_f]) for i, name in enumerate(feature_names)}
    return [item[0] for item in sorted(item_imp.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def run_ablation(mode='top_k', top_k=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_temp, X_static, y = np.load('data/processed/X_temporal_raw.npy'), np.load('data/processed/X_static_raw.npy'), np.load('data/processed/y.npy')
    f_names = np.load('data/processed/feature_names.npy')
    
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics, best_mae = [], float('inf')
    model_dir = 'results/models/'
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    print(f"\n--- Mod Başlatıldı: {mode.upper()} ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
        scaler = MinMaxScaler()
        N_tr, T, F = X_temp[train_idx].shape
        X_tr_s = scaler.fit_transform(X_temp[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
        X_val_s = scaler.transform(X_temp[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)

        curr_static_tr, curr_static_val = X_static[train_idx], X_static[val_idx]
        
        if mode == 'top_k':
            selected = select_features_in_fold(X_tr_s.reshape(N_tr, -1), y[train_idx], f_names, top_k)
            sel_idx = [i for i, name in enumerate(f_names) if name in selected]
            X_tr_f, X_val_f = X_tr_s[:, :, sel_idx], X_val_s[:, :, sel_idx]
        elif mode == 'static_only':
            X_tr_f, X_val_f = np.zeros((N_tr, T, 1)), np.zeros((len(val_idx), T, 1))
        elif mode == 'temporal_only':
            X_tr_f, X_val_f = X_tr_s, X_val_s
            curr_static_tr, curr_static_val = np.zeros((N_tr, 2)), np.zeros((len(val_idx), 2))

        model = TemporalTransformer(X_tr_f.shape[2], 2, d_model=128, nhead=8, num_layers=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001); criterion = nn.HuberLoss()
        
        train_loader = DataLoader(MIMICDataset(X_tr_f, curr_static_tr, y[train_idx]), batch_size=32, shuffle=True)
        val_loader = DataLoader(MIMICDataset(X_val_f, curr_static_val, y[val_idx]), batch_size=32)

        for epoch in range(30):
            model.train()
            for xt, xs, target in train_loader:
                xt, xs, target = xt.to(device), xs.to(device), target.to(device)
                optimizer.zero_grad(); criterion(model(xt, xs), target).backward(); optimizer.step()

        m = evaluate_all_metrics(model, val_loader, device)
        fold_metrics.append(m)
        
        if m['MAE'] < best_mae:
            best_mae = m['MAE']
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_transformer_{mode}.pth'))
    
    df = pd.DataFrame(fold_metrics).mean().to_frame().T
    df['Ablation_Mode'] = mode
    return df

if __name__ == "__main__":
    results = [run_ablation(mode=m) for m in ['top_k', 'static_only', 'temporal_only']]
    final = pd.concat(results)
    final.to_csv('results/transformer_results/ablation_metrics.csv', index=False)
    print("\n" + "="*40 + "\nABLATION STUDY TAMAMLANDI\n" + "="*40); print(final[['Ablation_Mode', 'MAE', 'R2']])