import os
import pandas as pd
import numpy as np
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnvV3
from rl_eth_engine.features.build_features import build_features
from rl_eth_engine.data.data_collector import DataCollector
from rl_eth_engine.evaluation.generate_v3_report import run_evaluation_report

class ConsistenciaCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ConsistenciaCallback, self).__init__(verbose)
        self.episode_trades = 0

    def _on_step(self) -> bool:
        # Extract info from the environment
        info = self.locals['infos'][0]
        if info.get('last_trade_pnl', 0) != 0:
            self.logger.record('trade/pnl', info['last_trade_pnl'])
            self.logger.record('trade/regime', info['last_trade_regime'])
        
        self.logger.record('env/equity', info['equity'])
        self.logger.record('env/trade_count', info['trade_count'])
        return True

def run_v3_industrial():
    print("Initiating V3 Industrial Grade RL Pipeline...")
    symbol = 'ETHUSDT'
    collector = DataCollector()
    
    # 1. Load Data (Ensuring we have the full 2020-2026 range)
    df_5m = collector.download_and_save(symbol, '5m', '1 Jan, 2020', '1 Jan, 2026')
    df_15m = collector.download_and_save(symbol, '15m', '1 Jan, 2020', '1 Jan, 2026')
    df_1h = collector.download_and_save(symbol, '1h', '1 Jan, 2020', '1 Jan, 2026')
    
    full_df = build_features(df_5m, df_15m, df_1h)
    
    # Seeds
    seeds = [11, 23, 37, 71, 97]
    
    # Windows
    windows = [
        {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-06-30')},
        {'train': ('2022-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-06-30')},
    ]

    for seed in seeds:
        print(f"\n===== RUNNING SEED {seed} =====")
        for i, win in enumerate(windows):
            print(f"--- WINDOW {i+1} ---")
            train_df = full_df.loc[win['train'][0]:win['train'][1]]
            
            # --- PHASE 1: Entry Focus (Semi-Exit) ---
            print("--- Training Phase 1: Entry focus ---")
            env = ETHTradingEnvV3(train_df, training_phase=1)
            env = Monitor(env)
            venv = DummyVecEnv([lambda: env])
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
            
            model = RecurrentPPO(
                "MlpLstmPolicy", venv, verbose=1, seed=seed,
                learning_rate=1e-4, n_steps=2048, batch_size=256,
                gamma=0.995, gae_lambda=0.97, clip_range=0.15,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                tensorboard_log="./tensorboard_logs/v3_industrial"
            )
            
            model.learn(total_timesteps=500000, tb_log_name=f"S{seed}_W{i+1}_P1", callback=ConsistenciaCallback())
            
            # --- PHASE 2: Management enabled ---
            print("--- Training Phase 2: Management focus ---")
            env_p2 = ETHTradingEnvV3(train_df, training_phase=2)
            env_p2 = Monitor(env_p2)
            venv_p2 = DummyVecEnv([lambda: env_p2])
            venv_p2 = VecNormalize(venv_p2, norm_obs=True, norm_reward=True)
            # Transfer stats
            venv_p2.obs_rms = venv.obs_rms
            
            model.set_env(venv_p2)
            model.learn(total_timesteps=500000, tb_log_name=f"S{seed}_W{i+1}_P2", reset_num_timesteps=False, callback=ConsistenciaCallback())
            
            # Save
            model_path = f"models/v3_industrial_S{seed}_W{i+1}"
            model.save(model_path)
            venv_p2.save(f"models/v3_stats_S{seed}_W{i+1}.pkl")
            
            # --- Evaluation & Stress Test ---
            print(f"--- Evaluating Window {i+1} (Multi-Stress) ---")
            test_df = full_df.loc[win['test'][0]:win['test'][1]]
            
            results = []
            for stress in [1.0, 1.5, 2.0]:
                print(f"Running Stress Test x{stress}...")
                m = run_evaluation_report(model_path, f"models/v3_stats_S{seed}_W{i+1}.pkl", test_df, cost_multiplier=stress)
                results.append(m)
                
            # Log results to CSV
            pdf = pd.DataFrame(results)
            pdf['seed'] = seed
            pdf['window'] = i+1
            report_path = "evaluation/v3_industrial_results.csv"
            pdf.to_csv(report_path, mode='a', header=not os.path.exists(report_path), index=False)
            print(f"Results logged to {report_path}")
            
if __name__ == "__main__":
    run_v3_industrial()
