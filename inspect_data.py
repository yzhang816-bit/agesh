import pandas as pd

try:
    df = pd.read_csv('sp500_raw_data.csv')
    print(f"Unique symbols: {df['Symbol'].unique()}")
    print(f"Number of symbols: {len(df['Symbol'].unique())}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
except Exception as e:
    print(f"Error: {e}")
