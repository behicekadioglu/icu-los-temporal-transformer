import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler



def load_filtered_csv(path, id_col, target_ids, cols):
    # Reading in chunks to filter relevant rows based on target IDs
    chunks = []
    for chunk in pd.read_csv(path, chunksize=1000000, usecols=cols):
        filtered = chunk[chunk[id_col].isin(target_ids)]
        if not filtered.empty:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)



def run_preprocessing(raw_path, processed_path):
    # Main Cohort Extraction
    icustays = pd.read_csv(os.path.join(raw_path, 'icu/icustays.csv.gz'))
    icustays = icustays[['subject_id', 'hadm_id', 'stay_id', 'intime', 'los']]

    patients = pd.read_csv(os.path.join(raw_path, 'hosp/patients.csv.gz'))
    patients = patients[['subject_id', 'gender', 'anchor_age']]

    df_main = pd.merge(icustays, patients, on='subject_id', how='inner')
    df_main['intime'] = pd.to_datetime(df_main['intime'])
    
    target_stay_ids = set(df_main['stay_id'].unique())
    target_hadm_ids = set(df_main['hadm_id'].unique())


    # Loading Clinical Data (Chart & Lab)
    df_all_charts = load_filtered_csv(os.path.join(raw_path, 'icu/chartevents.csv.gz'), 
                                      'stay_id', target_stay_ids, ['stay_id', 'charttime', 'itemid', 'valuenum'])
    
    df_all_labs = load_filtered_csv(os.path.join(raw_path, 'hosp/labevents.csv.gz'), 
                                    'hadm_id', target_hadm_ids, ['hadm_id', 'charttime', 'itemid', 'valuenum'])


    # Combining and Time Alignment
    df_all_labs = pd.merge(df_all_labs, df_main[['hadm_id', 'stay_id', 'intime']], on='hadm_id', how='left')
    df_all_charts = pd.merge(df_all_charts, df_main[['stay_id', 'intime']], on='stay_id', how='left')
    
    df_all_labs['itemid'] = 'L' + df_all_labs['itemid'].astype(str)
    df_all_charts['itemid'] = 'C' + df_all_charts['itemid'].astype(str)

    df_combined = pd.concat([df_all_charts, df_all_labs], ignore_index=True)
    df_combined['charttime'] = pd.to_datetime(df_combined['charttime'])
    df_combined['hour'] = ((df_combined['charttime'] - df_combined['intime']).dt.total_seconds() / 3600).astype(int)
    
    df_24h = df_combined[(df_combined['hour'] >= 0) & (df_combined['hour'] < 24)].copy()


    # Pivoting, Imputation, and Normalization
    df_pivot = df_24h.pivot_table(index=['stay_id', 'hour'], columns='itemid', values='valuenum', aggfunc='mean')
    full_index = pd.MultiIndex.from_product([df_main['stay_id'].unique(), range(24)], names=['stay_id', 'hour'])
    df_imputed = df_pivot.reindex(full_index).groupby('stay_id').ffill().fillna(df_pivot.mean()).reset_index()
 
    feature_cols = [c for c in df_imputed.columns if c not in ['stay_id', 'hour']]
    df_imputed[feature_cols] = MinMaxScaler().fit_transform(df_imputed[feature_cols])

    df_main['gender_encoded'] = df_main['gender'].map({'M': 1, 'F': 0})
    df_main['age_scaled'] = df_main['anchor_age'] / 100.0


    # Tensor Preparation (N=140, T=24, F=675+2)
    X_temporal_np = df_imputed[feature_cols].values.reshape(len(df_main), 24, -1)
    X_static_np = df_main[['gender_encoded', 'age_scaled']].values
    y = df_main['los'].values


    # Flattening for Baseline Models
    X_flattened = X_temporal_np.reshape(len(df_main), -1)
    X_baseline = np.hstack([X_flattened, X_static_np])


    # Save Processed Data
    if not os.path.exists(processed_path): os.makedirs(processed_path)
    np.save(os.path.join(processed_path, 'X_temporal.npy'), X_temporal_np)
    np.save(os.path.join(processed_path, 'X_static.npy'), X_static_np)
    np.save(os.path.join(processed_path, 'X_baseline.npy'), X_baseline)
    np.save(os.path.join(processed_path, 'y.npy'), y)
    print(f"Ön işleme tamamlandı! Özellik Sayısı: {len(feature_cols)}")


if __name__ == "__main__":
    run_preprocessing("data/raw/mimic-iv-clinical-database-demo-2.2", "data/processed/")