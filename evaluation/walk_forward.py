import os
import pandas as pd
import numpy as np
import yaml
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnv
from rl_eth_engine.features.build_features import build_features
from rl_eth_engine.data.data_collector import DataCollector
from sb3_contrib import RecurrentPPO
import datetime

def run_full_walk_forward():
    symbol = 'ETHUSDT'
    collector = DataCollector()
    
    # 1. Download/Load Data (Long range)
    print("Step 1: Downloading/Loading 2020-2026 data...")
    # For the sake of time in this task, I'll use a representative range if not already present
    # But the logic is set for the full range.
    df_raw_5m = collector.download_and_save(symbol, '5m', '1 Jan, 2020', '1 Jan, 2026')
    df_raw_15m = collector.download_and_save(symbol, '15m', '1 Jan, 2020', '1 Jan, 2026')
    df_raw_1h = collector.download_and_save(symbol, '1h', '1 Jan, 2020', '1 Jan, 2026')
    
    print("Step 2: building features...")
    df = build_features(df_raw_5m, df_raw_15m, df_raw_1h)
    
    # 2. Define Walk-Forward Windows
    # Train: 2 years, Val: 1 year, Test: 1 year
    windows = [
        {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},
        {'train': ('2021-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},
        {'train': ('2022-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},
        {'train': ('2023-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-12-31')},
    ]
    
    all_results = []
    
    for i, window in enumerate(windows):
        print(f"\n--- Walk-Forward Window {i+1}: Train {window['train']} | Test {window['test']} ---")
        
        train_df = df.loc[window['train'][0]:window['train'][1]]
        test_df = df.loc[window['test'][0]:window['test'][1]]
        
        if len(train_df) < 1000 or len(test_df) < 200:
            print(f"Skipping window {i+1} due to insufficient data.")
            continue
            
        # 3. Train
        env_train = ETHTradingEnv(train_df)
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env_train, 
            verbose=0,
            n_steps=1024,
            batch_size=128,
            learning_rate=0.0003,
            gamma=0.99
        )
        
        print(f"Training on {len(train_df)} steps...")
        model.learn(total_timesteps=50000) # Increased steps for real training
        
        # 4. Evaluate
        env_test = ETHTradingEnv(test_df)
        obs, _ = env_test.reset()
        
        test_pnl = 0
        equities = []
        
        print(f"Testing on {len(test_df)} steps...")
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_test.step(action)
            equities.append(info['equity'])
            
        final_equity = info['equity']
        net_return = (final_equity - 1000) / 1000
        max_dd = (max(equities) - min(equities)) / max(equities) if equities else 0
        
        result = {
            'window': i+1,
            'period': f"{window['test'][0]} - {window['test'][1]}",
            'net_return': net_return,
            'max_drawdown': max_dd,
            'final_equity': final_equity
        }
        all_results.append(result)
        print(f"Window {i+1} Result: Return={net_return:.2%}, MaxDD={max_dd:.2%}")
        
    return all_results

if __name__ == "__main__":
    results = run_full_walk_forward()
    print("\n--- FINAL WALK-FORWARD SUMMARY ---")
    for r in results:
        print(f"Window {r['window']} ({r['period']}): Return {r['net_return']:.2%}, MaxDD {r['max_drawdown']:.2%}")
