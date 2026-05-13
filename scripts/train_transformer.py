import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import optuna
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

class MIMICDataset(Dataset):
    def __init__(self, X_temporal, X_static, y):
        self.X_temporal = torch.tensor(X_temporal, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_temporal[idx], self.X_static[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, static_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model + static_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    def forward(self, x_temp, x_static):
        x = self.input_projection(x_temp)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x_combined = torch.cat([x, x_static], dim=1)
        return self.regression_head(x_combined).squeeze(-1)

def evaluate_all_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for x_t, x_s, y in dataloader:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            preds = model(x_t, x_s)
            all_preds.extend(preds.cpu().numpy())
            all_y.extend(y.cpu().numpy())
    all_preds, all_y = np.array(all_preds), np.array(all_y)
    return {'MAE': mean_absolute_error(all_y, all_preds), 'RMSE': np.sqrt(mean_squared_error(all_y, all_preds)), 'R2': r2_score(all_y, all_preds), 'MedAE': median_absolute_error(all_y, all_preds)}

def objective(trial, X_temp, X_static, y, device):
    m_params = {'d_model': trial.suggest_categorical('d_model', [64, 128, 256]), 'nhead': trial.suggest_categorical('nhead', [4, 8]), 'num_layers': trial.suggest_int('num_layers', 1, 3), 'dropout': trial.suggest_float('dropout', 0.1, 0.3)}
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    maes = []
    for train_idx, val_idx in kf.split(X_temp):
        scaler = MinMaxScaler()
        N_tr, T, F = X_temp[train_idx].shape
        X_tr_s = scaler.fit_transform(X_temp[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
        X_val_s = scaler.transform(X_temp[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)
        train_loader = DataLoader(MIMICDataset(X_tr_s, X_static[train_idx], y[train_idx]), batch_size=32, shuffle=True)
        val_loader = DataLoader(MIMICDataset(X_val_s, X_static[val_idx], y[val_idx]), batch_size=32)
        model = TemporalTransformer(F, X_static.shape[1], **m_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss()
        for epoch in range(15):
            model.train()
            for xt, xs, target in train_loader:
                xt, xs, target = xt.to(device), xs.to(device), target.to(device)
                optimizer.zero_grad()
                criterion(model(xt, xs), target).backward(); optimizer.step()
        maes.append(evaluate_all_metrics(model, val_loader, device)['MAE'])
    return np.mean(maes)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_temp, X_static, y = np.load('data/processed/X_temporal_raw.npy'), np.load('data/processed/X_static_raw.npy'), np.load('data/processed/y.npy')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_temp, X_static, y, device), n_trials=10)
    
    best_all = study.best_params.copy(); best_lr = best_all.pop('lr')
    
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    final_metrics, best_mae = [], float('inf')
    model_dir = 'results/models/'
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
        scaler = MinMaxScaler()
        N_tr, T, F = X_temp[train_idx].shape
        X_tr_s = scaler.fit_transform(X_temp[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
        X_val_s = scaler.transform(X_temp[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)
        train_loader = DataLoader(MIMICDataset(X_tr_s, X_static[train_idx], y[train_idx]), batch_size=32, shuffle=True)
        val_loader = DataLoader(MIMICDataset(X_val_s, X_static[val_idx], y[val_idx]), batch_size=32)
        
        model = TemporalTransformer(F, X_static.shape[1], **best_all).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_lr)
        criterion = nn.HuberLoss()
        
        for epoch in range(30):
            model.train()
            for xt, xs, target in train_loader:
                xt, xs, target = xt.to(device), xs.to(device), target.to(device)
                optimizer.zero_grad(); criterion(model(xt, xs), target).backward(); optimizer.step()
        
        metrics = evaluate_all_metrics(model, val_loader, device)
        final_metrics.append(metrics)
        
        # En iyi fold'u kaydet
        if metrics['MAE'] < best_mae:
            best_mae = metrics['MAE']
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_transformer_main.pth'))
            print(f"Fold {fold} en iyi model olarak kaydedildi. MAE: {best_mae:.3f}")

    df_res = pd.DataFrame(final_metrics).mean().to_frame().T
    df_res.to_csv('results/transformer_results/final_metrics.csv', index=False)
    print("\nFinal Sonuçlar Kaydedildi."); print(df_res)