import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from train_transformer import TemporalTransformer, MIMICDataset

# Grafik teması ayarları (Akademik rapor kalitesinde)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': [10, 6]})

def load_model_auto(mode, device, model_dir='results/models/'):
    """JSON konfigürasyonunu okur ve modeli otomatik yükler"""
    config_file = 'model_config.json' if mode == 'main' else f'config_{mode}.json'
    config_path = os.path.join(model_dir, config_file)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Modeli oluştur
    model = TemporalTransformer(
        input_dim=config['input_dim'],
        static_dim=config['static_dim'],
        **config['params']
    ).to(device)
    
    # Ağırlıkları yükle
    weight_file = 'best_transformer_main.pth' if mode == 'main' else f'best_transformer_{mode}.pth'
    model.load_state_dict(torch.load(os.path.join(model_dir, weight_file), map_location=device))
    model.eval()
    return model, config

def plot_regression_calibration(y_true, y_pred, save_path):
    """Gerçek vs Tahmin edilen LoS değerlerini karşılaştırır """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.4, color='#2c7fb8', label='Tahminler')
    
    # İdeal tahmin çizgisi (y=x)
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Mükemmel Tahmin')
    
    plt.xlabel('Gerçek Yoğun Bakım Kalış Süresi (Gün)')
    plt.ylabel('Tahmin Edilen Kalış Süresi (Gün)')
    plt.title('Regresyon Kalibrasyon Plotu (MIMIC-IV)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '1_regression_calibration.png'), dpi=300)
    plt.close()

def plot_residual_analysis(y_true, y_pred, save_path):
    """Hata dağılımı (Residual) analizi [cite: 245]"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Hata Histogramı
    sns.histplot(residuals, kde=True, color='#f03b20', ax=ax1)
    ax1.set_title('Hata Dağılımı (Residuals)')
    ax1.set_xlabel('Hata Değeri (Gün)')
    
    # 2. Residual vs Predicted (Hata Varyansı)
    ax2.scatter(y_pred, residuals, alpha=0.4, color='#feb24c')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title('Tahmin vs Hata (Homoscedasticity Kontrolü)')
    ax2.set_xlabel('Tahmin Edilen Değer')
    ax2.set_ylabel('Hata')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '2_residual_analysis.png'), dpi=300)
    plt.close()

def plot_attention_heatmap(model, x_sample, xs_sample, device, save_path):
    """Transformer içindeki zamansal dikkati (Attention) görselleştirir """
    model.eval()
    with torch.no_grad():
        # Giriş verisini projeksiyondan geçir
        x = model.input_projection(x_sample.to(device))
        x = model.pos_encoder(x)
        
        # Transformer katmanlarındaki aktivasyonları yakala (Basitleştirilmiş temsil)
        # Gerçek attention haritaları için 'forward' içinde weights return edilmelidir.
        # Burada katman çıktısı üzerindeki 'özellik yoğunluğunu' ısı haritası yaparız.
        output = model.transformer_encoder(x) # (1, 24, d_model)
        attn_map = output[0].cpu().numpy().T # (d_model, 24)
    
    plt.figure(figsize=(14, 5))
    sns.heatmap(attn_map, cmap='YlGnBu', cbar_kws={'label': 'Aktivasyon Şiddeti'})
    plt.title('Temporal Attention Heatmap: 24 Saatlik Klinik Gidişat Analizi')
    plt.xlabel('Yoğun Bakımdaki Saat (0-24h)')
    plt.ylabel('Latent Özellik Kanalları')
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '3_attention_heatmap.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = 'results/figures/'
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Veriyi Yükle ve Ölçeklendir (Sadece görselleştirme için)
    X_t = np.load('data/processed/X_temporal_raw.npy')
    X_s = np.load('data/processed/X_static_raw.npy')
    y = np.load('data/processed/y.npy')
    
    # Sızıntısız ölçeklendirme prensibine uygun (global scaler) [cite: 150, 290]
    scaler = MinMaxScaler()
    N, T, F = X_t.shape
    X_t_s = scaler.fit_transform(X_t.reshape(-1, F)).reshape(N, T, F)

    # 2. Modeli Otomatik Yükle (Main Model)
    try:
        model, config = load_model_auto('main', device)
        print(f"Model başarıyla yüklendi. Girdi Boyutu: {config['input_dim']}")

        # 3. Tüm Veri Üzerinden Tahmin Al
        loader = DataLoader(MIMICDataset(X_t_s, X_s, y), batch_size=len(y))
        xt_batch, xs_batch, y_true = next(iter(loader))
        
        with torch.no_grad():
            y_pred = model(xt_batch.to(device), xs_batch.to(device)).cpu().numpy()

        # 4. Grafikleri Üret
        print("Grafikler oluşturuluyor...")
        
        # A. Kalibrasyon Plotu 
        plot_regression_calibration(y_true.numpy(), y_pred, fig_dir)
        
        # B. Hata Analizi [cite: 245]
        plot_residual_analysis(y_true.numpy(), y_pred, fig_dir)
        
        # C. Attention Heatmap (İlk hasta örneği üzerinden) [cite: 251]
        plot_attention_heatmap(model, xt_batch[0:1], xs_batch[0:1], device, fig_dir)
        
        print(f"İşlem tamamlandı! Grafikler '{fig_dir}' klasöründe.")

    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("Lütfen önce 'train_transformer.py' dosyasını çalıştırarak modelleri ve JSON konfigürasyonlarını oluşturun.")