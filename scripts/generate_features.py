import os
import sys
import pandas as pd

# Add the parent directory of rl_eth_engine to sys.path
# Current script is in rl_eth_engine/scripts/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
parent_dir = os.path.abspath(os.path.join(project_root, '..'))
sys.path.append(parent_dir)

from rl_eth_engine.features.build_features import build_features

def main():
    print("Loading raw data...")
    raw_dir = "data/raw"
    df_5m = pd.read_parquet(os.path.join(raw_dir, "ETHUSDT_5m_2020_01_01_2026_01_01.parquet"))
    df_15m = pd.read_parquet(os.path.join(raw_dir, "ETHUSDT_15m_2020_01_01_2026_01_01.parquet"))
    df_1h = pd.read_parquet(os.path.join(raw_dir, "ETHUSDT_1h_2020_01_01_2026_01_01.parquet"))

    print("Building features (V3 Industrial Grade)...")
    # Set index to timestamp for reindexing logic in build_features
    df_5m.set_index('open_time', inplace=True)
    df_15m.set_index('open_time', inplace=True)
    df_1h.set_index('open_time', inplace=True)
    
    processed_df = build_features(df_5m, df_15m, df_1h)

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "features.parquet")
    processed_df.to_parquet(out_path)
    
    print(f"Features saved to: {out_path}")
    print(f"Shape: {processed_df.shape}")

if __name__ == "__main__":
    main()
