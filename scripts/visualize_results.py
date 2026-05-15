import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from train_transformer import TemporalTransformer, MIMICDataset

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})


def load_any_model(mode, device, model_dir='results/models/'):
    config_file = 'model_config.json' if mode == 'main' else f'config_{mode}.json'
    weight_file = 'best_transformer_main.pth' if mode == 'main' else f'best_transformer_{mode}.pth'
    
    path_cfg = os.path.join(model_dir, config_file)
    if not os.path.exists(path_cfg):
        return None, None

    with open(path_cfg, 'r') as f:
        config = json.load(f)
    
    model = TemporalTransformer(config['input_dim'], config['static_dim'], **config['params']).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, weight_file), map_location=device))
    model.eval()

    return model, config



def plot_individual_results(mode, y_true, y_pred, save_dir):
    # Regression Calibration Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='#2c7fb8')
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Guess')
    plt.xlabel('Actual Length of Stay (Days)')
    plt.ylabel('Predicted Length of Stay (Days)')
    plt.title(f'Regression Calibration - {mode.upper()}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'calibration_{mode}.png'), dpi=300)
    plt.close()


    # Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, color='#f03b20')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Residual Distribution - {mode.upper()}')
    plt.xlabel('Residual (Days)')
    plt.savefig(os.path.join(save_dir, f'residuals_{mode}.png'), dpi=300)
    plt.close()



def get_predictions(model, mode, X_t, X_s, y, device):
    N, T, F = X_t.shape
    
    if mode == 'static_only':
        xt_in, xs_in = np.zeros((N, T, 1)), X_s

    elif mode == 'temporal_only':
        xt_in, xs_in = X_t, np.zeros((N, 2))

    elif mode == 'top_k':
        k = model.input_projection.in_features
        xt_in, xs_in = X_t[:, :, :k], X_s

    else:
        xt_in, xs_in = X_t, X_s

    loader = torch.utils.data.DataLoader(MIMICDataset(xt_in, xs_in, y), batch_size=len(y))
    xt_b, xs_b, y_b = next(iter(loader))
    
    with torch.no_grad():
        preds = model(xt_b.to(device), xs_b.to(device)).cpu().numpy()

    return y_b.numpy(), preds



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = 'results/figures/'
    os.makedirs(fig_dir, exist_ok=True)

    X_t = np.load('data/processed/X_temporal_raw.npy')
    X_s = np.load('data/processed/X_static_raw.npy')
    y = np.load('data/processed/y.npy')
    
    modes = ['main', 'top_k', 'static_only', 'temporal_only']
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    summary_data = []

    for i, m in enumerate(modes):
        model, cfg = load_any_model(m, device)
        if model is None:
            print(f"{m} model not found, skipping...")
            continue
            
        print(f"Graphics are being generated for: {m}")
        y_true, y_pred = get_predictions(model, m, X_t, X_s, y, device)
        
        plot_individual_results(m, y_true, y_pred, fig_dir)
        
        # Scatter plot for direct comparison
        axes[i].scatter(y_true, y_pred, alpha=0.4, color='#2c7fb8')
        axes[i].plot([0, 20], [0, 20], 'r--', lw=2)
        axes[i].set_title(f'MOD: {m.upper()}')
        axes[i].set_xlabel('Actual Length of Stay (Days)')
        if i == 0: axes[i].set_ylabel('Predicted Length of Stay (Days)')
        
        mae = np.mean(np.abs(y_true - y_pred))
        summary_data.append({'Mode': m, 'MAE': mae})


    plt.suptitle('Ablation Study Comparison (Day Scale)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, 'ablation_grid_all.png'), dpi=300)
    
    # MAE Summary Plot
    if summary_data:
        df_mae = pd.DataFrame(summary_data)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_mae, x='Mode', y='MAE', palette='viridis')
        plt.title('MAE Comparison Across Models')
        plt.savefig(os.path.join(fig_dir, 'ablation_mae_summary.png'), dpi=300)
    
    print(f"\nDone! All plots saved to '{fig_dir}' directory.")