import os
import sys
import pandas as pd
import numpy as np
import yaml

# Inject project root into sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Also add parent of project root for package-style imports
parent_dir = os.path.abspath(os.path.join(project_root, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

def update_status(gate, seed, window, step=0, total_steps=0, stage="training"):
    os.makedirs("logs", exist_ok=True)
    status = {
        "gate": gate,
        "seed": seed,
        "window": window,
        "step": step,
        "total_steps": total_steps,
        "stage": stage,
        "last_update": pd.Timestamp.now().isoformat()
    }
    with open("logs/status.json", "w") as f:
        json.dump(status, f, indent=4)
    
    # Heartbeat for quick terminal check
    with open("logs/heartbeat.txt", "w") as f:
        f.write(f"{pd.Timestamp.now().isoformat()} | G{gate} | S{seed} | W{window} | Step {step}/{total_steps}")

class V4LoggerCallback(BaseCallback):
    def __init__(self, gate, seed, window, total_steps, verbose=0):
        super(V4LoggerCallback, self).__init__(verbose)
        self.gate = gate
        self.seed = seed
        self.window = window
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if info.get('last_trade_pnl', 0) != 0:
            self.logger.record('trade/pnl', info['last_trade_pnl'])
            self.logger.record('trade/regime', info['last_trade_regime'])
        self.logger.record('env/equity', info['equity'])
        self.logger.record('env/trade_count', info['trade_count'])
        
        # Rule 6: 4-Layer Monitoring - Update status every 1024 steps
        if self.num_timesteps % 1024 == 0:
            update_status(self.gate, self.seed, self.window, self.num_timesteps, self.total_steps)
        return True

def get_git_metadata():
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return commit, status
    except:
        return "N/A", "N/A"

def get_config_hash(path):
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return "N/A"

def run_v4_gated_pipeline(gate=1):
    import torch
    print(f"--- INITIATING V4 GATE {gate} ---")
    
    # Rule 4 & 10: GPU Enforcement for long runs
    cuda_available = torch.cuda.is_available()
    print(f"GPU Available: {cuda_available}")
    if gate >= 2 and not cuda_available:
        print("ERROR: Gate 2+ requires GPU runtime. Aborting.")
        with open("logs/error.flag", "w") as f: f.write("GPU Missing for Industrial run")
        return

    # Rule 2: Sync check
    commit, status = get_git_metadata()
    if status != "":
        print(f"ERROR: Repository is dirty. Status:\n{status}")
        with open("logs/error.flag", "w") as f: f.write("Dirty Repository")
        return

    os.makedirs("logs", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    symbol = 'ETHUSDT'
    collector = DataCollector()
    
    # Gate configurations (Rule 9: Industrial criteria)
    gate_configs = {
        1: {'seeds': [11, 23, 37], 'steps': 300000, 'windows': 1, 'thresholds': {'trades': 50, 'expectancy': -0.03, 'flat_ratio': 0.90}},
        2: {'seeds': [11, 23, 37, 71, 97], 'steps': 800000, 'windows': 2, 'thresholds': {'trades': 100, 'pf': 1.05, 'expectancy': 0.0}}
    }
    cfg = gate_configs[gate]
    
    # Data loading
    df_5m = collector.download_and_save(symbol, '5m', '1 Jan, 2020', '1 Jan, 2024')
    df_15m = collector.download_and_save(symbol, '15m', '1 Jan, 2020', '1 Jan, 2024')
    df_1h = collector.download_and_save(symbol, '1h', '1 Jan, 2020', '1 Jan, 2024')
    full_df = build_features(df_5m, df_15m, df_1h)
    
    dataset_version = "v3_industrial_2020_2024"
    config_path = os.path.join(project_root, 'configs', 'reward_config.yaml')
    config_hash = get_config_hash(config_path)
    
    windows = [
        {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},
        {'train': ('2022-01-01', '2023-06-30'), 'test': ('2023-07-01', '2023-12-31')}
    ]
    
    results = []
    try:
        for seed in cfg['seeds']:
            for win_idx in range(cfg['windows']):
                win = windows[win_idx]
                win_id = win_idx + 1
                update_status(gate, seed, win_id, stage="initializing")
                
                print(f"\n>> Gate {gate} | Seed {seed} | Window {win_id}")
                
                model_path = f"models/v4_G{gate}_S{seed}_W{win_id}"
                stats_path = f"models/v4_stats_G{gate}_S{seed}_W{win_id}.pkl"
                
                train_df = full_df.loc[win['train'][0]:win['train'][1]]
                test_df = full_df.loc[win['test'][0]:win['test'][1]]
                
                if not os.path.exists(f"{model_path}.zip"):
                    env = ETHTradingEnvV4(train_df, training_phase=1)
                    env = Monitor(env)
                    venv = DummyVecEnv([lambda: env])
                    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
                    
                    model = RecurrentPPO(
                        "MlpLstmPolicy", venv, verbose=1, seed=seed,
                        learning_rate=1e-4, n_steps=2048, batch_size=128,
                        tensorboard_log="./tensorboard_logs/v4_industrial"
                    )
                    
                    model.learn(
                        total_timesteps=cfg['steps'], 
                        tb_log_name=f"V4_G{gate}_S{seed}_W{win_id}", 
                        callback=V4LoggerCallback(gate, seed, win_id, cfg['steps'])
                    )
                    model.save(model_path)
                    venv.save(stats_path)
                
                # Rule 3: Comprehensive Audit Metadata
                for stress in [1.0, 1.5]:
                    update_status(gate, seed, win_id, stage=f"eval_stress_{stress}")
                    m = run_evaluation_report(model_path, stats_path, test_df, cost_multiplier=stress)
                    m.update({
                        'commit_hash': commit,
                        'dataset_version': dataset_version,
                        'config_hash': config_hash,
                        'seed': seed,
                        'window_id': win_id,
                        'gate': gate
                    })
                    results.append(m)
                    
                # Incremental save
                pd.DataFrame(results).to_csv("evaluation/industrial_results.csv", index=False)

        # Gate Passage Check
        pdf = pd.DataFrame(results)
        avg_trades = pdf[pdf['cost_stress'] == 1.0]['trade_count'].mean()
        avg_pf = pdf[pdf['cost_stress'] == 1.0]['profit_factor'].mean()
        avg_expectancy = pdf[pdf['cost_stress'] == 1.0]['expectancy'].mean()
        
        print(f"\n--- GATE {gate} FINAL REPORT ---")
        print(f"Avg Trades: {avg_trades:.2f} | Avg PF: {avg_pf:.2f} | Avg Expectancy: {avg_expectancy:.4f}R")
        
        summary = {
            'gate': gate,
            'avg_trades': avg_trades,
            'avg_pf': avg_pf,
            'avg_expectancy': avg_expectancy,
            'passed': False,
            'commit': commit,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if gate == 1:
            if avg_trades > cfg['thresholds']['trades'] and avg_expectancy > cfg['thresholds']['expectancy']:
                summary['passed'] = True
        elif gate == 2:
            if avg_pf >= cfg['thresholds']['pf'] and avg_expectancy >= cfg['thresholds']['expectancy']:
                summary['passed'] = True
                
        with open("evaluation/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        if summary['passed']:
            print(f"CONGRATULATIONS: GATE {gate} PASSED.")
            with open("logs/done.flag", "w") as f: f.write(f"Gate {gate} Success")
        else:
            print(f"CRITICAL: GATE {gate} FAILED KPIs.")
            with open("logs/error.flag", "w") as f: f.write(f"Gate {gate} KPI Failure")

    except Exception as e:
        print(f"CRITICAL ERROR in pipeline: {e}")
        with open("logs/error.flag", "w") as f: f.write(str(e))
        raise e

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, default=1)
    args = parser.parse_args()
    
    run_v4_gated_pipeline(gate=args.gate)
