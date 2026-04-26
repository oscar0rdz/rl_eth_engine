import os
import pandas as pd
import numpy as np
import yaml
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnv
from rl_eth_engine.features.build_features import build_features
from rl_eth_engine.data.data_collector import DataCollector
from rl_eth_engine.training.curriculum_trainer_v2 import CurriculumTrainerV2

def run_v2_walk_forward():
    print("Initiating V2 Rolling Walk-Forward Training...")
    symbol = 'ETHUSDT'
    collector = DataCollector()
    
    # Complete Data (already downloaded)
    df_5m = collector.download_and_save(symbol, '5m', '1 Jan, 2020', '1 Jan, 2026')
    df_15m = collector.download_and_save(symbol, '15m', '1 Jan, 2020', '1 Jan, 2026')
    df_1h = collector.download_and_save(symbol, '1h', '1 Jan, 2020', '1 Jan, 2026')
    
    full_df = build_features(df_5m, df_15m, df_1h)
    
    # Windows: Train 24m / Val 6m / Test 6m
    # Example Window: 2021-2022 (Train), 2023 H1 (Val), 2023 H2 (Test)
    # We'll do 3 rolling windows
    windows = [
        {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-06-30')},
        {'train': ('2021-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-06-30')},
        {'train': ('2022-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-06-30')},
    ]
    
    all_results = []
    
    for i, win in enumerate(windows):
        print(f"\n--- WINDOW {i+1}: Train {win['train']} | Test {win['test']} ---")
        train_df = full_df.loc[win['train'][0]:win['train'][1]]
        test_df = full_df.loc[win['test'][0]:win['test'][1]]
        
        # 1. Curriculum Training on Window i
        trainer = CurriculumTrainerV2(train_df, log_dir=f"./tensorboard_logs/v2_win_{i+1}")
        
        # Stage A (0.25x friction) - 300k
        model, venv = trainer.run_stage(stage_name="Etapa_A", friction=0.25, steps=200000)
        # Stage B (0.5x friction) - 500k
        model, venv = trainer.run_stage(model=model, stage_name="Etapa_B", friction=0.5, steps=300000)
        # Stage C (1.0x friction) - 500k
        model, venv = trainer.run_stage(model=model, stage_name="Etapa_C", friction=1.0, steps=500000)
        
        # 2. Evaluation on Test Set
        # Separate evaluation env with frozen VecNormalize stats
        eval_env = ETHTradingEnv(test_df)
        eval_env = Monitor(eval_env)
        eval_venv = DummyVecEnv([lambda: eval_env])
        # Load stats from training
        eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_venv.obs_rms = venv.obs_rms
        eval_venv.training = False # DO NOT update stats during test
        
        mean_reward, std_reward = evaluate_policy(model, eval_venv, n_eval_episodes=1)
        
        # Extract metrics from eval_env (via Monitor)
        # info contains equity and pos
        res = {
            'window': i+1,
            'mean_reward': mean_reward,
            'final_equity': eval_env.equity,
            'trades': eval_env.trades_count if hasattr(eval_env, 'trades_count') else 0
        }
        all_results.append(res)
        print(f"Window {i+1} Result: Final Equity={eval_env.equity:.2f}")

    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("v2_walk_forward_results.csv")
    print("\n--- V2 WALK-FORWARD COMPLETE ---")
    print(results_df)

if __name__ == "__main__":
    run_v2_walk_forward()
