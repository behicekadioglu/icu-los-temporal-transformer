import pandas as pd
import numpy as np
import os
import xgboost as xgb

SEED = 22

def load_filtered_csv(path, id_col, target_ids, cols):
    """
    Reads large MIMIC-IV CSV files in chunks and filters rows 
    based on the cohort's stay/admission IDs.
    """
    chunks = []

    for chunk in pd.read_csv(path, chunksize=1000000, usecols=cols):
        filtered = chunk[chunk[id_col].isin(target_ids)]

        if not filtered.empty:
            chunks.append(filtered)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)



def run_preprocessing(raw_path, processed_path, top_k=15):
    print("Preprocessing started: ")
    print("Loading cohort data...")
    
    # Cohort Extraction
    icustays = pd.read_csv(os.path.join(raw_path, 'icu/icustays.csv.gz'))[['subject_id', 'hadm_id', 'stay_id', 'intime', 'los']]
    patients = pd.read_csv(os.path.join(raw_path, 'hosp/patients.csv.gz'))[['subject_id', 'gender', 'anchor_age']]

    df_main = pd.merge(icustays, patients, on='subject_id', how='inner')
    df_main['intime'] = pd.to_datetime(df_main['intime'])
    
    target_stay_ids = set(df_main['stay_id'].unique())
    target_hadm_ids = set(df_main['hadm_id'].unique())


    # Getting Item Dictionaries for Mapping IDs to Clinical Names
    print("Loading item dictionaries...")
    d_items = pd.read_csv(os.path.join(raw_path, 'icu/d_items.csv.gz'))[['itemid', 'label']]
    d_items['itemid'] = 'C - ' + d_items['itemid'].astype(str)
    
    d_labitems = pd.read_csv(os.path.join(raw_path, 'hosp/d_labitems.csv.gz'))[['itemid', 'label']]
    d_labitems['itemid'] = 'L - ' + d_labitems['itemid'].astype(str)
    
    item_mapping = pd.concat([d_items, d_labitems]).set_index('itemid')['label'].to_dict()


    # Load Clinical Events
    print("Loading clinical events...")
    df_all_charts = load_filtered_csv(os.path.join(raw_path, 'icu/chartevents.csv.gz'), 
                                      'stay_id', target_stay_ids, ['stay_id', 'charttime', 'itemid', 'valuenum'])
    
    df_all_labs = load_filtered_csv(os.path.join(raw_path, 'hosp/labevents.csv.gz'), 
                                    'hadm_id', target_hadm_ids, ['hadm_id', 'charttime', 'itemid', 'valuenum'])


    # Time Alignment and Merging
    df_all_labs = pd.merge(df_all_labs, df_main[['hadm_id', 'stay_id', 'intime']], on='hadm_id', how='left')
    df_all_charts = pd.merge(df_all_charts, df_main[['stay_id', 'intime']], on='stay_id', how='left')
    
    # Prefix item IDs to distinguish between Lab and Chart events
    df_all_labs['itemid'] = 'L - ' + df_all_labs['itemid'].astype(str)
    df_all_charts['itemid'] = 'C - ' + df_all_charts['itemid'].astype(str)

    df_combined = pd.concat([df_all_charts, df_all_labs], ignore_index=True)
    df_combined['charttime'] = pd.to_datetime(df_combined['charttime'])
    
    # Calculate the hour from admission (intime)
    df_combined['hour'] = ((df_combined['charttime'] - df_combined['intime']).dt.total_seconds() / 3600).astype(int)
    
    # Filter for the first 24 hours of the ICU stay
    df_24h = df_combined[(df_combined['hour'] >= 0) & (df_combined['hour'] < 24)].copy()


    # Pivoting and Imputation
    print("Pivoting data and handling missing values...")
    df_pivot = df_24h.pivot_table(index=['stay_id', 'hour'], columns='itemid', values='valuenum', aggfunc='mean')
    
    # Ensure every patient has all 24 hours represented
    full_index = pd.MultiIndex.from_product([df_main['stay_id'].unique(), range(24)], names=['stay_id', 'hour'])
    df_imputed = df_pivot.reindex(full_index).groupby('stay_id').ffill().fillna(df_pivot.mean()).reset_index()

    # Define feature columns
    feature_cols = [c for c in df_imputed.columns if c not in ['stay_id', 'hour']]

    # Static Feature Preparation (Raw)
    df_main['gender_encoded'] = df_main['gender'].map({'M': 1, 'F': 0})
    df_main['age_raw'] = df_main['anchor_age']
    
    # Raw Tensor
    X_temporal_raw = df_imputed[feature_cols].values.reshape(len(df_main), 24, -1)
    X_static_raw = df_main[['gender_encoded', 'age_raw']].values
    y = df_main['los'].values


    # Feature Selection
    print("Performing feature selection with XGBoost...")
    X_flat = X_temporal_raw.reshape(len(df_main), -1)
    
    fs_model = xgb.XGBRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
    fs_model.fit(X_flat, y)
    
    importances = fs_model.feature_importances_
    num_features = len(feature_cols)
    item_importances = {}
    
    for i, col_id in enumerate(feature_cols):
        # Calculate total importance for each feature across all time steps
        total_imp = sum(importances[i::num_features])

        # Map item ID to clinical name, if available, otherwise keep the ID
        actual_name = item_mapping.get(col_id, col_id) 
        item_importances[col_id] = {'name': actual_name, 'score': total_imp}
        
    # Sort features by importance score in descending order
    sorted_items = sorted(item_importances.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Select top K features based on importance scores
    selected_ids = [item[0] for item in sorted_items[:top_k]]
    selected_names = [item[1]['name'] for item in sorted_items[:top_k]]
    selected_scores = [item[1]['score'] for item in sorted_items[:top_k]]
    
    # Filter the temporal data to keep only the selected features
    selected_indices = [feature_cols.index(col_id) for col_id in selected_ids]
    X_temporal_selected = X_temporal_raw[:, :, selected_indices]


    # Save Processed Data
    if not os.path.exists(processed_path): 
        os.makedirs(processed_path)
    
    np.save(os.path.join(processed_path, 'X_temporal_raw.npy'), X_temporal_raw)
    np.save(os.path.join(processed_path, 'X_temporal_selected.npy'), X_temporal_selected)
    np.save(os.path.join(processed_path, 'X_static_raw.npy'), X_static_raw)
    np.save(os.path.join(processed_path, 'y.npy'), y)
    

    # Save selected feature names and their importance scores for reference
    np.save(os.path.join(processed_path, 'selected_feature_names.npy'), np.array(selected_names))
    np.save(os.path.join(processed_path, 'selected_feature_scores.npy'), np.array(selected_scores))
    
    
    print(f"Preprocessing completed! Saved top {top_k} features globally.")
    print("Selected clinical parameters:", selected_names)



if __name__ == "__main__":
    # IMPORTANT: Update path to match your local structure
    run_preprocessing("data/raw/mimic-iv-clinical-database-demo-2.2", "data/processed/")