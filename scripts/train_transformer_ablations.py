import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from train_transformer import TemporalTransformer, MIMICDataset, evaluate_all_metrics

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

def select_features_in_fold(X_train_flat, y_train, feature_names, top_k):
    """train_baselines.py dosyasındaki özellik seçimi mantığı"""
    fs_model = xgb.XGBRegressor(n_estimators=50, random_state=SEED)
    fs_model.fit(X_train_flat, y_train)
    importances = fs_model.feature_importances_
    num_features = len(feature_names)
    item_importances = {}
    for i, name in enumerate(feature_names):
        total_imp = sum(importances[i::num_features])
        item_importances[name] = total_imp
    sorted_items = sorted(item_importances.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:top_k]]

def run_ablation_study(mode='top_k', top_k=30):
    """
    mode seçenekleri: 
    'top_k': Özellik seçimi uygulanmış tam model 
    'static_only': Sadece yaş ve cinsiyet 
    'temporal_only': Sadece klinik zaman serisi 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = 'data/processed/'
    X_temp = np.load(os.path.join(data_path, 'X_temporal_raw.npy'))
    X_static = np.load(os.path.join(data_path, 'X_static_raw.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    f_names = np.load(os.path.join(data_path, 'feature_names.npy'))

    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    all_results = []

    print(f"\n--- Ablasyon Başlatıldı: {mode.upper()} ---")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
        # 1. Sızıntısız Ölçeklendirme
        scaler = MinMaxScaler()
        N_tr, T, F = X_temp[train_idx].shape
        X_tr_scaled = scaler.fit_transform(X_temp[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
        X_val_scaled = scaler.transform(X_temp[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)

        # 2. Mod Seçimine Göre Veri Filtreleme
        current_X_static_tr = X_static[train_idx]
        current_X_static_val = X_static[val_idx]
        
        if mode == 'top_k':
            # XGBoost ile özellik seçimi 
            selected_items = select_features_in_fold(X_tr_scaled.reshape(N_tr, -1), y[train_idx], f_names, top_k)
            sel_idx = [i for i, name in enumerate(f_names) if name in selected_items]
            X_tr_final = X_tr_scaled[:, :, sel_idx]
            X_val_final = X_val_scaled[:, :, sel_idx]
        elif mode == 'static_only':
            X_tr_final = torch.zeros((N_tr, T, 1)) # Boş temporal girdi
            X_val_final = torch.zeros((len(val_idx), T, 1))
        elif mode == 'temporal_only':
            X_tr_final = X_tr_scaled
            X_val_final = X_val_scaled
            current_X_static_tr = np.zeros((N_tr, 2)) # Statik veriyi sıfırla
            current_X_static_val = np.zeros((len(val_idx), 2))

        # 3. Model Kurulumu ve Eğitim (Önceki en iyi parametreleri kullanabilirsin)
        input_dim = X_tr_final.shape[2]
        model = TemporalTransformer(input_dim, 2, d_model=128, nhead=8, num_layers=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.HuberLoss()

        train_loader = DataLoader(MIMICDataset(X_tr_final, current_X_static_tr, y[train_idx]), batch_size=32, shuffle=True)
        val_loader = DataLoader(MIMICDataset(X_val_final, current_X_static_val, y[val_idx]), batch_size=32)

        for epoch in range(30):
            model.train()
            for xt, xs, target in train_loader:
                xt, xs, target = xt.to(device), xs.to(device), target.to(device)
                optimizer.zero_grad()
                criterion(model(xt, xs), target).backward()
                optimizer.step()

        metrics = evaluate_all_metrics(model, val_loader, device)
        all_results.append(metrics)
        print(f"Fold {fold} MAE: {metrics['MAE']:.3f}")

    # Sonuçları Kaydet
    df_res = pd.DataFrame(all_results).mean().to_frame().T
    df_res['Ablation_Mode'] = mode
    return df_res

if __name__ == "__main__":
    # Tüm senaryoları sırayla çalıştır
    results_list = []
    for m in ['top_k', 'static_only', 'temporal_only']:
        results_list.append(run_ablation_study(mode=m))
    
    final_report = pd.concat(results_list)
    print("\n" + "="*40 + "\nABLATION STUDY SONUÇLARI\n" + "="*40)
    print(final_report[['Ablation_Mode', 'MAE', 'R2']])
    final_report.to_csv('results/transformer_results/ablation_study_results.csv', index=False)