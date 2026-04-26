import os
import pandas as pd
import yaml
from sb3_contrib import RecurrentPPO
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnv
from rl_eth_engine.features.build_features import build_features
from rl_eth_engine.data.data_collector import DataCollector

def train_recurrent_ppo(symbol='ETHUSDT', total_timesteps=100000):
    # 1. Load Config
    with open('rl_eth_engine/configs/reward_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Fetch/Prepare Data
    collector = DataCollector()
    print(f"Preparing data for {symbol}...")
    # Fetching 3 months of data for a quick demo check
    df_5m = collector.download_and_save(symbol, '5m', '3 months ago UTC')
    df_15m = collector.download_and_save(symbol, '15m', '3 months ago UTC')
    df_1h = collector.download_and_save(symbol, '1h', '3 months ago UTC')
    
    # 3. Build Features
    print("Building features...")
    df = build_features(df_5m, df_15m, df_1h)
    
    # 4. Initialize Env
    print("Initializing environment...")
    env = ETHTradingEnv(df)
    
    # 5. Initialize Model
    print("Initializing Recurrent PPO (LSTM)...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1,
        n_steps=1024,
        batch_size=128,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 6. Train
    print(f"Starting training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    
    # 7. Save
    save_path = "rl_eth_engine/models/recurrent_ppo/eth_recurrent_ppo_v1"
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_recurrent_ppo()
