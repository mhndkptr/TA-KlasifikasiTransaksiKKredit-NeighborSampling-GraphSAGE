import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

def load_and_preprocess_data(file_path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load the IBM Credit Card Transactions dataset and perform basic EDA/preprocessing.
    Converts timestamps, sorts chronologically, cleans numeric fields, and encodes categoricals.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, nrows=nrows)

    print("Parsing timestamps...")
    # Fill missing time with 00:00 to avoid parsing errors
    df['Time'] = df['Time'].fillna('00:00')
    df['datetime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str).str.zfill(2) + '-' + 
        df['Day'].astype(str).str.zfill(2) + ' ' + 
        df['Time']
    )
    
    print("Sorting chronologically...")
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print("Cleaning Amount...")
    df['Amount'] = df['Amount'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    
    print("Encoding target label...")
    df['Is Fraud?'] = df['Is Fraud?'].map({'Yes': 1, 'No': 0})
    
    print("Encoding categorical features...")
    cat_columns = ['Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?']
    for col in cat_columns:
        df[col] = df[col].fillna('<missing>')
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    # Scale Amount
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Extract numerical features for Transaction node
    feature_cols = ['Amount'] + cat_columns
    
    return df, feature_cols

def temporal_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    """
    Split the dataset based on temporal order.
    """
    n_samples = len(df)
    train_end = int(train_frac * n_samples)
    val_end = int((train_frac + val_frac) * n_samples)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

