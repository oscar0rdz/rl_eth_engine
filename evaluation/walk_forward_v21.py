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
from rl_eth_engine.evaluation.baseline_strategy import evaluate_baseline

def run_v21_walk_forward():
    print("Initiating V2.1 Rolling Walk-Forward & Baseline Comparison...")
    symbol = 'ETHUSDT'
    collector = DataCollector()
    
    # 1. Load Data
    df_5m = collector.download_and_save(symbol, '5m', '1 Jan, 2020', '1 Jan, 2026')
    df_15m = collector.download_and_save(symbol, '15m', '1 Jan, 2020', '1 Jan, 2026')
    df_1h = collector.download_and_save(symbol, '1h', '1 Jan, 2020', '1 Jan, 2026')
    
    full_df = build_features(df_5m, df_15m, df_1h)
    
    # Windows: Train 24m / Test 6m
    windows = [
        {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-06-30')},
        {'train': ('2021-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-06-30')},
        {'train': ('2022-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-06-30')},
    ]
    
    summary_report = []
    
    for i, win in enumerate(windows):
        print(f"\n--- WINDOW {i+1}: Train {win['train']} | Test {win['test']} ---")
        train_df = full_df.loc[win['train'][0]:win['train'][1]]
        test_df = full_df.loc[win['test'][0]:win['test'][1]]
        
        # --- Rama A: RL V2.1 ---
        trainer = CurriculumTrainerV2(train_df, log_dir=f"./tensorboard_logs/v21_win_{i+1}")
        # Shorter steps for the check
        model, venv = trainer.run_stage(stage_name="Etapa_A", friction=0.25, steps=100000)
        model, venv = trainer.run_stage(model=model, stage_name="Etapa_B", friction=0.5, steps=200000)
        model, venv = trainer.run_stage(model=model, stage_name="Etapa_C", friction=1.0, steps=300000)
        
        # Eval RL
        eval_env = ETHTradingEnv(test_df)
        eval_env = Monitor(eval_env)
        eval_venv = DummyVecEnv([lambda: eval_env])
        eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False)
        eval_venv.obs_rms = venv.obs_rms
        eval_venv.training = False
        
        evaluate_policy(model, eval_venv, n_eval_episodes=1)
        rl_equity = eval_env.equity
        rl_trades = eval_env.trades_count if hasattr(eval_env, 'trades_count') else 0
        
        # --- Rama B: Baseline ---
        print(f"Evaluating Baseline on Test Window {i+1}...")
        base_equity, base_trades = evaluate_baseline(test_df, ETHTradingEnv)
        
        # --- Benchmark: Buy & Hold ---
        bh_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0]) - 1
        bh_equity = 1000 * (1 + bh_return)
        
        print(f"Window {i+1} Results:")
        print(f"  RL Equity: {rl_equity:.2f}")
        print(f"  Baseline Equity: {base_equity:.2f}")
        print(f"  Buy&Hold Equity: {bh_equity:.2f}")
        
        summary_report.append({
            'window': i+1,
            'rl_equity': rl_equity,
            'rl_trades': rl_trades,
            'base_equity': base_equity,
            'base_trades': base_trades,
            'bh_equity': bh_equity
        })

    # Final Summary
    report_df = pd.DataFrame(summary_report)
    report_df.to_csv("v21_comparison_results.csv")
    print("\n--- V2.1 WALK-FORWARD COMPLETE ---")
    print(report_df)

if __name__ == "__main__":
    run_v21_walk_forward()
