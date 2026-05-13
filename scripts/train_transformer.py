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



# Reprodüksiyon için sabit seed (Progress Report'ta belirtilen standartlara uygun)
SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)





# 1. Dataset Sınıfı
class MIMICDataset(Dataset):
    def __init__(self, X_temporal, X_static, y):
        self.X_temporal = torch.tensor(X_temporal, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_temporal[idx], self.X_static[idx], self.y[idx]





# 2. Model Mimarisi (Positional Encoding + Transformer) [cite: 204, 205]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]





class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, static_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
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
        x = x.mean(dim=1) # Global Average Pooling
        x_combined = torch.cat([x, x_static], dim=1)
        return self.regression_head(x_combined).squeeze(-1)




# 3. Metrik Hesaplama Fonksiyonu [cite: 188]
def evaluate_all_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for x_t, x_s, y in dataloader:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            preds = model(x_t, x_s)
            all_preds.extend(preds.cpu().numpy())
            all_y.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_y = np.array(all_y)
    
    return {
        'MAE': mean_absolute_error(all_y, all_preds),
        'RMSE': np.sqrt(mean_squared_error(all_y, all_preds)),
        'R2': r2_score(all_y, all_preds),
        'MedAE': median_absolute_error(all_y, all_preds)
    }




# 4. Optuna ve K-Fold Eğitim Döngüsü
def objective(trial, X_temp, X_static, y, device):
    # Önerilen Hiperparametreler [cite: 209, 210]
    params = {
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'nhead': trial.suggest_categorical('nhead', [4, 8]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_maes = []

    for train_idx, val_idx in kf.split(X_temp):
        # Sızıntısız Ölçeklendirme (Leakage-Free Scaling) [cite: 150, 290]
        scaler = MinMaxScaler()
        N_tr, T, F = X_temp[train_idx].shape
        X_tr_scaled = scaler.fit_transform(X_temp[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
        X_val_scaled = scaler.transform(X_temp[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)

        train_ds = MIMICDataset(X_tr_scaled, X_static[train_idx], y[train_idx])
        val_ds = MIMICDataset(X_val_scaled, X_static[val_idx], y[val_idx])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        model = TemporalTransformer(F, X_static.shape[1], **params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.HuberLoss(delta=1.0) # Aykırı değerlere karşı direnç 

        # Kısa eğitim (Optuna için hızlı deneme)
        for epoch in range(15):
            model.train()
            for x_t, x_s, target in train_loader:
                x_t, x_s, target = x_t.to(device), x_s.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_t, x_s), target)
                loss.backward()
                optimizer.step()

        metrics = evaluate_all_metrics(model, val_loader, device)
        fold_maes.append(metrics['MAE'])
        
    return np.mean(fold_maes)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim cihazı: {device}")

    # Verileri Yükle
    data_path = 'data/processed/'
    X_temp = np.load(os.path.join(data_path, 'X_temporal_raw.npy'))
    X_static = np.load(os.path.join(data_path, 'X_static_raw.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))

    # Optuna Çalışması
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_temp, X_static, y, device), n_trials=10)

    # Final Değerlendirme (En iyi parametrelerle tüm metrikleri alma)
    best_params = study.best_params
    print(f"\nEn iyi parametreler: {best_params}")


    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    final_fold_metrics = []



    print("\nFinal Modeli Eğitiliyor (10-Fold)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
        # Scaling
        scaler = MinMaxScaler()
        N_tr, T, F = X_temp[train_idx].shape
        X_tr_scaled = scaler.fit_transform(X_temp[train_idx].reshape(-1, F)).reshape(N_tr, T, F)
        X_val_scaled = scaler.transform(X_temp[val_idx].reshape(-1, F)).reshape(len(val_idx), T, F)

        train_loader = DataLoader(MIMICDataset(X_tr_scaled, X_static[train_idx], y[train_idx]), batch_size=32, shuffle=True)
        val_loader = DataLoader(MIMICDataset(X_val_scaled, X_static[val_idx], y[val_idx]), batch_size=32)

        model = TemporalTransformer(F, X_static.shape[1], **best_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
        criterion = nn.HuberLoss(delta=1.0)

        for epoch in range(30): # Final eğitim için daha uzun epoch
            model.train()
            for x_t, x_s, target in train_loader:
                x_t, x_s, target = x_t.to(device), x_s.to(device), target.to(device)
                optimizer.zero_grad()
                criterion(model(x_t, x_s), target).backward()
                optimizer.step()

        metrics = evaluate_all_metrics(model, val_loader, device)
        final_fold_metrics.append(metrics)
        print(f"Fold {fold} Tamamlandı. MAE: {metrics['MAE']:.3f}")



    # Sonuçları Tabloya Dönüştür ve Kaydet
    df_results = pd.DataFrame(final_fold_metrics).mean().to_frame().T
    df_results['Model'] = 'Temporal Transformer'
    
    res_path = 'results/transformer_results/'
    if not os.path.exists(res_path): os.makedirs(res_path)
    df_results.to_csv(os.path.join(res_path, 'transformer_final_metrics.csv'), index=False)
    
    print("\n" + "="*30)
    print("TRANSFORMER FINAL SONUÇLARI")
    print(df_results[['Model', 'MAE', 'RMSE', 'R2', 'MedAE']].to_string(index=False))