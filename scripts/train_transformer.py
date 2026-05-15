import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os, json, math, optuna, time
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

SEED = 22
torch.manual_seed(SEED)



class MIMICDataset(Dataset):
    def __init__(self, X_t, X_s, y):
        self.X_t = torch.tensor(X_t, dtype=torch.float32)
        self.X_s = torch.tensor(X_s, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, idx): 
        return self.X_t[idx], self.X_s[idx], self.y[idx]



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x): 
        return x + self.pe[:, :x.size(1)]



class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, static_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.regression_head = nn.Sequential(nn.Linear(d_model + static_dim, d_model // 2), 
                                             nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
    
    def forward(self, xt, xs):
        x = self.pos_encoder(self.input_projection(xt))
        x = self.transformer_encoder(x).mean(dim=1)
        return self.regression_head(torch.cat([x, xs], dim=1)).squeeze(-1)



def evaluate_metrics(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    start_inf = time.time()

    with torch.no_grad():
        for xt, xs, y in loader:
            all_p.extend(model(xt.to(device), xs.to(device)).cpu().numpy())
            all_y.extend(y.numpy())

    inf_time = (time.time() - start_inf) / len(loader.dataset)
    all_p, all_y = np.array(all_p), np.array(all_y)
    return {'MAE': mean_absolute_error(all_y, all_p), 
            'RMSE': np.sqrt(mean_squared_error(all_y, all_p)), 
            'R2': r2_score(all_y, all_p), 
            'MedAE': median_absolute_error(all_y, all_p), 
            'Inf_Time': inf_time}



def objective(trial, X_t, X_s, y, device):
    m_p = {'d_model': trial.suggest_categorical('d_model', [64, 128]), 'nhead': trial.suggest_categorical('nhead', [4, 8]), 
           'num_layers': trial.suggest_int('num_layers', 1, 3), 'dropout': trial.suggest_float('dropout', 0.1, 0.3)}
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    maes = []

    for tr_idx, val_idx in kf.split(X_t):
        scaler = MinMaxScaler()
        N, T, F = X_t[tr_idx].shape
        xt_tr = scaler.fit_transform(X_t[tr_idx].reshape(-1, F)).reshape(N, T, F)
        xt_val = scaler.transform(X_t[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)

        train_l = DataLoader(MIMICDataset(xt_tr, X_s[tr_idx], y[tr_idx]), batch_size=32, shuffle=True)
        val_l = DataLoader(MIMICDataset(xt_val, X_s[val_idx], y[val_idx]), batch_size=32)
        model = TemporalTransformer(F, X_s.shape[1], **m_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for e in range(10):
            model.train()
            for xt, xs, target in train_l:
                optimizer.zero_grad(); nn.HuberLoss()(model(xt.to(device), xs.to(device)), 
                                                      target.to(device)).backward(); optimizer.step()
    
        maes.append(evaluate_metrics(model, val_l, device)['MAE'])

    return np.mean(maes)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = np.load('data/processed/X_temporal_raw.npy') 
    X_s = np.load('data/processed/X_static_raw.npy')
    y = np.load('data/processed/y.npy')

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, X_t, X_s, y, device), n_trials=10)
    
    best_p = study.best_params.copy(); lr = best_p.pop('lr')
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    results, best_mae = [], float('inf')
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_t)):
        scaler = MinMaxScaler()
        N, T, F = X_t[tr_idx].shape

        xt_tr = scaler.fit_transform(X_t[tr_idx].reshape(-1, F)).reshape(N, T, F)
        xt_val = scaler.transform(X_t[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)

        train_l = DataLoader(MIMICDataset(xt_tr, X_s[tr_idx], y[tr_idx]), batch_size=32, shuffle=True)
        val_l = DataLoader(MIMICDataset(xt_val, X_s[val_idx], y[val_idx]), batch_size=32)
        model = TemporalTransformer(F, X_s.shape[1], **best_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        start_train = time.time()
        for e in range(20):
            model.train()
            for xt, xs, target in train_l:
                optimizer.zero_grad(); nn.HuberLoss()(model(xt.to(device), xs.to(device)), 
                                                      target.to(device)).backward(); optimizer.step()

        train_time = time.time() - start_train
        
        m = evaluate_metrics(model, val_l, device)
        m['Train_Time'] = train_time
        results.append(m)

        if m['MAE'] < best_mae:
            best_mae = m['MAE']
            os.makedirs('results/models', exist_ok=True)
            torch.save(model.state_dict(), 'results/models/best_transformer_main.pth')
            with open('results/models/model_config.json', 'w') as f: json.dump({'params': best_p, 'input_dim': F, 
                                                                                'static_dim': X_s.shape[1]}, f)
    

    
    print("\n" + "="*30 + "\nFINAL TRANSFORMER RESULTS\n" + "="*30)
    print(pd.DataFrame(results).mean())