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



def plot_global_feature_importance(data_dir='data/processed/', save_dir='results/figures/'):
    """
    Reads the globally selected feature names and scores from the preprocessing 
    stage and generates a horizontal bar chart.
    """
    names_path = os.path.join(data_dir, 'selected_feature_names.npy')
    scores_path = os.path.join(data_dir, 'selected_feature_scores.npy')
    
    if not (os.path.exists(names_path) and os.path.exists(scores_path)):
        print("Feature importance files not found. Skipping importance plot.")
        return

    features = np.load(names_path)
    scores = np.load(scores_path)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=scores, y=features, hue=features, legend=False, palette="viridis")
    plt.title('Top Selected Clinical Parameters (Global Importance)')
    plt.xlabel('Aggregated XGBoost Feature Importance')
    plt.ylabel('Clinical Parameter')
    plt.tight_layout()

    file_name = os.path.join(save_dir, 'global_feature_importance.png')
    plt.savefig(file_name)
    plt.close()
    print(f"Global feature importance plot saved to: {file_name}")



def plot_los_distribution(y_data, save_dir):
    """
    Plots the distribution of the target variable (Length of Stay).
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y_data, kde=True, bins=50, color='#2ca02c')
    plt.title('Distribution of Length of Stay (LoS)')
    plt.xlabel('Length of Stay (Days)')
    plt.ylabel('Frequency (Number of Patients)')
    plt.tight_layout()
    
    file_name = os.path.join(save_dir, 'los_distribution.png')
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"LoS distribution plot saved to: {file_name}")



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
 
    else: # main model
        xt_in, xs_in = X_t, X_s

    loader = torch.utils.data.DataLoader(MIMICDataset(xt_in, xs_in, y), batch_size=len(y))
    xt_b, xs_b, y_b = next(iter(loader))
    
    with torch.no_grad():
        preds = model(xt_b.to(device), xs_b.to(device)).cpu().numpy()

    return y_b.numpy(), preds



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = 'results/figures/'
    data_dir = 'data/processed/'
    os.makedirs(fig_dir, exist_ok=True)

    plot_global_feature_importance(data_dir, fig_dir)

    X_t_path = os.path.join(data_dir, 'X_temporal_selected.npy')
    if os.path.exists(X_t_path):
        X_t = np.load(X_t_path)
        X_s = np.load(os.path.join(data_dir, 'X_static_raw.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        
        plot_los_distribution(y, fig_dir)
        
        modes = ['main', 'static_only', 'temporal_only']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
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
            sns.barplot(data=df_mae, x='Mode', y='MAE', palette='viridis', hue='Mode', legend=False)
            plt.title('MAE Comparison Across Models')
            plt.savefig(os.path.join(fig_dir, 'ablation_mae_summary.png'), dpi=300)
    else:
        print(f"Temporal data not found at {X_t_path}. Make sure preprocessing has completed.")
        
    print(f"\nDone! All plots saved to '{fig_dir}' directory.")