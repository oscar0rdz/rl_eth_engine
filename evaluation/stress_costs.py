import pandas as pd
import numpy as np
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnv
from sb3_contrib import RecurrentPPO


def run_stress_test(model_path, df, multipliers=[1.0, 1.5, 2.0]):
    """
    Runs stress test by increasing costs (fees, slippage).
    """
    model = RecurrentPPO.load(model_path)
    results = {}
    
    for mult in multipliers:
        print(f"Running stress test with cost multiplier: {mult}x")
        
        # Modify env parameters for stress
        env = ETHTradingEnv(df)
        env.config['env_params']['trading_fee'] *= mult
        env.config['env_params']['slippage'] *= mult
        
        obs, _ = env.reset()
        equities = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            equities.append(info['equity'])
            
        final_equity = info['equity']
        results[mult] = {
            'final_equity': final_equity,
            'return': (final_equity - 1000) / 1000,
            'max_dd': (max(equities) - min(equities)) / max(equities) if equities else 0
        }
        print(f"Multiplier {mult}x: Return={results[mult]['return']:.2%}, MaxDD={results[mult]['max_dd']:.2%}")
        
    return results

if __name__ == "__main__":
    # This would be called after walk_forward.py
    pass

