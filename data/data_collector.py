import os
import pandas as pd
from binance.client import Client
from datetime import datetime

class DataCollector:
    def __init__(self, raw_dir='data/raw'):
        # Try to find .env in project root
        api_key = None
        api_secret = None
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "Clave API:" in line:
                        api_key = line.split(":", 1)[1].strip()
                    elif "Clave secreta:" in line:
                        api_secret = line.split(":", 1)[1].strip()
        
        self.client = Client(api_key, api_secret)
        self.raw_dir = raw_dir
        os.makedirs(raw_dir, exist_ok=True)


    def download_and_save(self, symbol, interval, start_str, end_str=None):
        # Support both CSV and Parquet extensions for loading
        csv_filename = f"{symbol}_{interval}_{start_str.replace(' ', '_')}.csv"
        parquet_filename = f"{symbol}_{interval}_2020_01_01_2026_01_01.parquet" # Match download script
        
        csv_path = os.path.join(self.raw_dir, csv_filename)
        parquet_path = os.path.join(self.raw_dir, parquet_filename)
        
        if os.path.exists(parquet_path):
            print(f"File {parquet_filename} already exists. Loading Parquet...")
            df = pd.read_parquet(parquet_path)
            # Ensure it has a datetime index if requested
            if 'open_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['open_time'])
                df.set_index('timestamp', inplace=True)
            return df

        if os.path.exists(csv_path):
            print(f"File {csv_filename} already exists. Loading CSV...")
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)

        print(f"Downloading {symbol} {interval} from {start_str}...")
        klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        df[numeric_cols].to_csv(csv_path)
        print(f"Saved to {csv_path}")
        return df[numeric_cols]

if __name__ == "__main__":
    collector = DataCollector()
    # Demo fetch
    # collector.download_and_save('ETHUSDT', Client.KLINE_INTERVAL_5MINUTE, "1 Jan, 2024", "1 Feb, 2024")
