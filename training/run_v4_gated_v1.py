import os
import pandas as pd
import numpy as np
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnvV4
from rl_eth_engine.features.build_features import build_features
from rl_eth_engine.data.data_collector import DataCollector
from rl_eth_engine.evaluation.generate_v3_report import run_evaluation_report

class V4LoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(V4LoggerCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if info.get('last_trade_pnl', 0) != 0:
            self.logger.record('trade/pnl', info['last_trade_pnl'])
            self.logger.record('trade/regime', info['last_trade_regime'])
        self.logger.record('env/equity', info['equity'])
        self.logger.record('env/trade_count', info['trade_count'])
        return True

def run_v4_gate_1():
    print("--- INITIATING V4 GATE 1 (Activity Validation) ---")
    symbol = 'ETHUSDT'
    collector = DataCollector()
    df_5m = collector.download_and_save(symbol, '5m', '1 Jan, 2020', '1 Jan, 2023')
    df_15m = collector.download_and_save(symbol, '15m', '1 Jan, 2020', '1 Jan, 2023')
    df_1h = collector.download_and_save(symbol, '1h', '1 Jan, 2020', '1 Jan, 2023')
    full_df = build_features(df_5m, df_15m, df_1h)
    
    seeds = [11, 23, 37] # 3 seeds for Gate 1
    win = {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')}
    train_df = full_df.loc[win['train'][0]:win['train'][1]]
    test_df = full_df.loc[win['test'][0]:win['test'][1]]
    
    # Evaluation Settings for Gate 1 (Stochastic to capture activity)
    eval_deterministic = False # Per User Request to confirm pattern learning
    cost_stresses = [1.0, 1.5]
    
    results = []
    for seed in seeds:
        print(f"\n>> Seed {seed} | Phase A (Entry Focus)")
        model_path = f"models/v4_G1_S{seed}"
        stats_path = f"models/v4_stats_G1_S{seed}.pkl"
        
        if not os.path.exists(f"{model_path}.zip"):
            env = ETHTradingEnvV4(train_df, training_phase=1)
            env = Monitor(env)
            venv = DummyVecEnv([lambda: env])
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
            
            model = RecurrentPPO(
                "MlpLstmPolicy", venv, verbose=1, seed=seed,
                learning_rate=3e-4, n_steps=2048, batch_size=128,
                clip_range=0.2, ent_coef=0.01,
                tensorboard_log="./tensorboard_logs/v4_industrial"
            )
            
            # 300k steps for Gate 1
            model.learn(total_timesteps=300000, tb_log_name=f"V4_G1_S{seed}", callback=V4LoggerCallback())
            model.save(model_path)
            venv.save(stats_path)
        else:
            print(f"Skipping training for Seed {seed}, model exists.")

        # Evaluation with stochastic inference for activity validation
        for stress in cost_stresses:
            print(f"Eval Seed {seed} (Stress {stress}x)...")
            m = run_evaluation_report(model_path, stats_path, test_df, cost_multiplier=stress, deterministic=eval_deterministic)
            m['seed'] = seed
            m['is_stochastic'] = not eval_deterministic
            results.append(m)
        
        # Iterative Save
        pd.DataFrame(results).to_csv("evaluation/v4_gate1_results.csv", index=False)
        
    # Check Gate 1 Passage
    pdf = pd.DataFrame(results)
    avg_trades = pdf[pdf['cost_stress'] == 1.0]['trade_count'].mean()
    if avg_trades > 50:
        print(f"\nGATE 1 PASSED (Avg Trades: {avg_trades}). Proceeding to Gate 2 (1.5M steps)...")
        # GATE 2 IMPLEMENTATION
        # [Implementation of Gate 2 logic would go here, which we can trigger manually or via this loop]
    else:
        print(f"\nGATE 1 FAILED or INCOMPLETE (Avg Trades: {avg_trades}).")

if __name__ == "__main__":
    run_v4_gate_1()
