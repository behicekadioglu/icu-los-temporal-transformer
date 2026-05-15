import torch
import os
import json
import numpy as np 
import pandas as pd
import xgboost as xgb
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from train_transformer import TemporalTransformer, MIMICDataset, evaluate_metrics

SEED = 22

def select_features(X_flat, y, f_names, top_k):
    fs = xgb.XGBRegressor(n_estimators=50, random_state=SEED).fit(X_flat, y)
    imp = {n: sum(fs.feature_importances_[i::len(f_names)]) for i, n in enumerate(f_names)}
    return [item[0] for item in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_k]]



def run_ablation(mode='top_k'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = np.load('data/processed/X_temporal_raw.npy')
    X_s = np.load('data/processed/X_static_raw.npy')
    y = np.load('data/processed/y.npy')
    f_names = np.load('data/processed/feature_names.npy')
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    results, best_mae = [], float('inf')
    m_params = {'d_model': 64, 'nhead': 4, 'num_layers': 2}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_t)):
        scaler = MinMaxScaler()
        N, T, F = X_t[tr_idx].shape
        xt_tr = scaler.fit_transform(X_t[tr_idx].reshape(-1, F)).reshape(N, T, F)
        xt_val = scaler.transform(X_t[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)
        xs_tr, xs_val = X_s[tr_idx], X_s[val_idx]

        if mode == 'top_k':
            sel = select_features(xt_tr.reshape(N, -1), y[tr_idx], f_names, 30)
            idx = [i for i, n in enumerate(f_names) if n in sel]
            xt_tr, xt_val = xt_tr[:, :, idx], xt_val[:, :, idx]
        elif mode == 'static_only': xt_tr, xt_val = np.zeros((N, T, 1)), np.zeros((len(val_idx), T, 1))
        elif mode == 'temporal_only': xs_tr, xs_val = np.zeros((N, 2)), np.zeros((len(val_idx), 2))

        model = TemporalTransformer(xt_tr.shape[2], xs_tr.shape[1], **m_params).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        
        start_train = time.time()
        for e in range(20):
            model.train()

            for xt, xs, target in DataLoader(MIMICDataset(xt_tr, xs_tr, y[tr_idx]), batch_size=32, shuffle=True):
                opt.zero_grad(); torch.nn.HuberLoss()(model(xt.to(device), xs.to(device)), target.to(device)).backward(); opt.step()

        train_time = time.time() - start_train
        
        m = evaluate_metrics(model, DataLoader(MIMICDataset(xt_val, xs_val, y[val_idx]), batch_size=32), device)
        m['Train_Time'] = train_time
        results.append(m)
        
        if m['MAE'] < best_mae:
            best_mae = m['MAE']
            torch.save(model.state_dict(), f'results/models/best_transformer_{mode}.pth')
            with open(f'results/models/config_{mode}.json', 'w') as f: json.dump({'params': m_params, 'input_dim': xt_tr.shape[2], 
                                                                                  'static_dim': xs_tr.shape[1]}, f)
    
    return pd.DataFrame(results).mean()



if __name__ == "__main__":
    for m in ['top_k', 'static_only', 'temporal_only']: 
        print(f"\n--- Ablation Mode: {m} ---")
        print(run_ablation(m))