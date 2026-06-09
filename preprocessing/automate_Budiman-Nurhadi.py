import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data, target_threshold=0.005):
    """
    Fungsi otomatisasi preprocessing data Bitcoin.
    Mengikuti struktur pipeline profesional dengan membagi data menjadi Train & Test set.
    
    Parameters:
    - data (DataFrame): Dataframe mentah dari btc_raw.csv
    - target_threshold (float): Threshold persentase untuk penentuan kelas target
    
    Returns:
    - X_train, X_test, y_train, y_test (arrays/DataFrames) yang siap dilatih oleh model
    """
    df_prep = data.copy()

    # --- 1. FEATURE ENGINEERING (Sama seperti eksperimen sebelumnya) ---
    df_prep['MA_7'] = df_prep['Close'].rolling(window=7).mean()
    df_prep['MA_21'] = df_prep['Close'].rolling(window=21).mean()
    df_prep['Daily_Return'] = df_prep['Close'].pct_change()
    df_prep['Close_Yesterday'] = df_prep['Close'].shift(1)
    df_prep['Volatility_7d'] = df_prep['Close'].rolling(window=7).std()

    # Perhitungan RSI_14
    delta = df_prep['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_prep['RSI_14'] = 100 - (100 / (1 + rs))

    # --- 2. MULTI-CLASS TARGET GENERATION ---
    df_prep['Next_Return'] = (df_prep['Close'].shift(-1) - df_prep['Close']) / df_prep['Close']
    
    conditions = [
        (df_prep['Next_Return'] > target_threshold),                  # Kelas 2
        (df_prep['Next_Return'] < -target_threshold),                 # Kelas 0
        (df_prep['Next_Return'].between(-target_threshold, target_threshold)) # Kelas 1
    ]
    df_prep['Target'] = np.select(conditions, [2, 0, 1], default=np.nan)

    # --- 3. CLEANING ---
    # Menghapus baris yang memiliki NaN akibat rolling/shift/diff
    df_clean = df_prep.dropna().copy()

    # --- 4. PEMISAHAN FITUR (X) DAN TARGET (y) ---
    # Menghapus kolom target, kolom bocor (Next_Return), dan kolom non-numerik (Date) jika ada
    drop_cols = ['Target', 'Next_Return']
    if 'Date' in df_clean.columns:
        drop_cols.append('Date')
        
    X = df_clean.drop(columns=drop_cols)
    y = df_clean['Target'].astype(int) # Memastikan target bertipe integer

    # --- 5. TRAIN-TEST SPLIT ---
    # Data dibagi menjadi data latih dan data uji sebelum proses Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- 6. PIPELINE & COLUMN TRANSFORMER ---
    # Menentukan fitur numerik yang akan di-scale
    numeric_features = X.columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    # --- 7. FITTING & TRANSFORMATION ---
    # Fit dan transform hanya pada training set
    X_train_scaled = preprocessor.fit_transform(X_train)
    
    # Transform pada testing set menggunakan parameter dari training set (mencegah data leakage)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test