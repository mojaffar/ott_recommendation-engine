import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    df = df[df['rating'] > 0]
    return df

def train_test_split(df):
    df = df.sort_values("timestamp")
    split_idx = int(0.8 * len(df))
    return df[:split_idx], df[split_idx:]