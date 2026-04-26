import os
import pandas as pd
import numpy as np
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnvV4
from rl_eth_engine.features.build_features import build_features
from rl_eth_engine.data.data_collector import DataCollector

def calculate_v3_metrics(equity_series, trade_history):
    """
    Expectancy, PF, Win/Loss, DD, etc.
    """
    if not trade_history:
        return {
            'net_return': 0.0, 'max_dd': 0.0, 'pf': 0.0, 
            'expectancy': 0.0, 'win_loss_ratio': 0.0, 'trade_count': 0
        }
    
    trades = pd.DataFrame(trade_history) # Expects list of dicts with 'pnl'
    wins = trades[trades['pnl'] > 0]['pnl']
    losses = trades[trades['pnl'] <= 0]['pnl']
    
    net_return = (equity_series[-1] / equity_series[0]) - 1
    max_equity = np.maximum.accumulate(equity_series)
    dd = (max_equity - equity_series) / max_equity
    max_dd = np.max(dd)
    
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
    expectancy = (net_return * 1000) / len(trades) # Simplistic expectancy per trade in base units
    
    return {
        'net_return': net_return,
        'max_dd': max_dd,
        'pf': pf,
        'expectancy': expectancy,
        'win_loss_ratio': avg_win / avg_loss,
        'trade_count': len(trades)
    }

def run_evaluation_report(model_path, stats_path, test_df, cost_multiplier=1.0, deterministic=True):
    """
    Run evaluation on a test set and generate V3 report with stress costs.
    """
    env = ETHTradingEnvV4(test_df, training_phase=2) 
    # Apply Stress Costs
    env.config['env_params']['trading_fee'] *= cost_multiplier
    env.config['env_params']['slippage'] *= cost_multiplier
    
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(stats_path, venv)
    venv.training = False
    venv.norm_reward = False
    
    model = RecurrentPPO.load(model_path, env=venv)
    obs = venv.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    equities = []
    done = False
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        equities.append(infos[0]['equity'])
        episode_starts = dones
        done = dones[0]
        
    # Final trade history from env
    trade_history = env.trade_history
    metrics = calculate_v3_metrics(np.array(equities), trade_history)
    metrics['cost_stress'] = cost_multiplier
    return metrics

if __name__ == "__main__":
    # Example placeholder
    print("V3 Reporter Ready.")
