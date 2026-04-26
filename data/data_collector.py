import os
import pandas as pd
from binance.client import Client
from datetime import datetime

class DataCollector:
    def __init__(self, raw_dir='rl_eth_engine/data/raw'):
        # Load keys from .env if possible
        api_key = None
        api_secret = None
        env_path = "/Users/oscarr/Desarrollo/Python0/PPO:LSTM + Gymnasium/.env"
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
        filename = f"{symbol}_{interval}_{start_str.replace(' ', '_')}.csv"
        filepath = os.path.join(self.raw_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists. Loading...")
            return pd.read_csv(filepath, index_col=0, parse_dates=True)

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
        
        df[numeric_cols].to_csv(filepath)
        print(f"Saved to {filepath}")
        return df[numeric_cols]

if __name__ == "__main__":
    collector = DataCollector()
    # Demo fetch
    # collector.download_and_save('ETHUSDT', Client.KLINE_INTERVAL_5MINUTE, "1 Jan, 2024", "1 Feb, 2024")
