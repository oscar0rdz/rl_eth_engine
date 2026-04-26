import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
import argparse

def download_klines(symbol, interval, start_str, end_str, out_dir):
    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
    LIMIT = 1500
    
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    os.makedirs(out_dir, exist_ok=True)
    
    all_rows = []
    cur = start_ts
    
    print(f"Downloading {symbol} {interval} from {start_str} to {end_str}...")
    
    while cur < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ts,
            "limit": LIMIT,
        }
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            
            all_rows.extend(rows)
            last_open_time = rows[-1][0]
            next_cur = last_open_time + 1
            if next_cur <= cur:
                break
            cur = next_cur
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    cols = [
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ]
    
    df = pd.DataFrame(all_rows, columns=cols)
    if df.empty:
        print("No data downloaded.")
        return
        
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    
    num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    
    filename = f"{symbol}_{interval}_{start_str.replace('-','_')}_{end_str.replace('-','_')}.parquet"
    path = os.path.join(out_dir, filename)
    df.to_parquet(path, index=False)
    
    print(f"Saved to: {path}")
    print(f"Total rows: {len(df)}")
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()
    
    download_klines(args.symbol, args.interval, args.start, args.end, args.out_dir)
