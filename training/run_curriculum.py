import os
import pandas as pd
from rl_eth_engine.data.data_collector import DataCollector
from rl_eth_engine.features.build_features import build_features
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnv
from rl_eth_engine.training.curriculum_trainer import CurriculumTrainer

def run_curriculum_main():
    symbol = 'ETHUSDT'
    collector = DataCollector()
    
    # 1. Prepare Data (if not already downloaded)
    print("Preparing data for curriculum training...")
    df_5m = collector.download_and_save(symbol, '5m', '1 Jan, 2020', '1 Jan, 2026')
    df_15m = collector.download_and_save(symbol, '15m', '1 Jan, 2020', '1 Jan, 2026')
    df_1h = collector.download_and_save(symbol, '1h', '1 Jan, 2020', '1 Jan, 2026')
    
    df = build_features(df_5m, df_15m, df_1h)
    
    # 2. Initialize Env
    env = ETHTradingEnv(df)
    trainer = CurriculumTrainer(env)
    
    # --- STAGE A: Survival ---
    model = trainer.train_stage_a(total_timesteps=300000)
    model.save("rl_eth_engine/models/recurrent_ppo/curriculum_stage_a")
    
    # --- STAGE B: Entries ---
    model = trainer.train_stage_b(model, total_timesteps=700000)
    model.save("rl_eth_engine/models/recurrent_ppo/curriculum_stage_b")
    
    # --- STAGE C: Robustness ---
    # We can pass model back to stage C
    model = trainer.train_stage_c(model, total_timesteps=1000000)
    model.save("rl_eth_engine/models/recurrent_ppo/curriculum_stage_c_final")
    
    print("\n--- CURRICULUM TRAINING COMPLETE ---")

if __name__ == "__main__":
    run_curriculum_main()
