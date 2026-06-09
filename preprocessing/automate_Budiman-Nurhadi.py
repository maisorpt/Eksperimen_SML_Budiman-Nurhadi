import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data, target_threshold=0.005):
    """
    Fungsi otomatisasi preprocessing data Bitcoin tanpa StandardScaler (untuk model berbasis Tree).
    Membagi data secara kronologis (shuffle=False) untuk mencegah data leakage waktu.
    
    Parameters:
    - data (DataFrame): Dataframe mentah yang dibaca dari btc_raw.csv
    - target_threshold (float): Threshold persentase untuk penentuan kelas target
    
    Returns:
    - X_train, X_test, y_train, y_test yang siap digunakan untuk training model.
    """
    df_prep = data.copy()

    # --- 1. FEATURE ENGINEERING ---
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

    # --- 2. MULTI-CLASS TARGET ---
    df_prep['Next_Return'] = (df_prep['Close'].shift(-1) - df_prep['Close']) / df_prep['Close']
    
    conditions = [
        (df_prep['Next_Return'] > target_threshold),                  # Kelas 2
        (df_prep['Next_Return'] < -target_threshold),                 # Kelas 0
        (df_prep['Next_Return'].between(-target_threshold, target_threshold)) # Kelas 1
    ]
    df_prep['Target'] = np.select(conditions, [2, 0, 1], default=np.nan)

    # --- 3. CLEANING ---
    df_clean = df_prep.dropna().copy()

    # Pisahkan Fitur (X) dan Target (y)
    X = df_clean.drop(['Target', 'Next_Return'], axis=1)
    if 'Date' in X.columns:
        X = X.drop(['Date'], axis=1)

    y = df_clean['Target'].astype(int)

    # --- 4. PENYIMPANAN DATA UTUH KE CSV ---
    df_final = X.copy()
    df_final['Target'] = y.values
    df_final.to_csv("btc_preprocessing.csv", index=True)
    print("  [INFO] File 'btc_preprocessing.csv' berhasil diperbarui dan disimpan.")

    # --- 5. TRAIN-TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df_raw = pd.read_csv("../btc_raw.csv")
    
    preprocess_data(df_raw)