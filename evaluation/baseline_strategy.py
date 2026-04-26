import pandas as pd
import numpy as np

class DeterministicBaseline:
    """
    Rama B: Strong Baseline No-RL covering 4 Edge Families
    """
    def __init__(self, df):
        self.df = df
        
    def generate_signals(self):
        """
        0=HOLD, 1=OPEN_S, 2=OPEN_F, 3=REDUCE, 4=CLOSE, 5=ADD
        """
        signals = np.zeros(len(self.df))
        
        for i in range(1, len(self.df)):
            row = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # --- Familia 1: Trend Pullback ---
            # HTF Trend UP + Pullback to EMA20
            if row['htf_trend'] > 0.0001 and row['dist_vwap'] < 0 and row['body_pct'] > 0.5:
                signals[i] = 1 # OPEN_SMALL
                
            # --- Familia 2: Breakout Continuation ---
            # EMA Slope + Change in Price > Threshold
            if row['ema20_slope'] > 0.0005 and row['ret_3'] > 0.002:
                signals[i] = 2 # OPEN_FULL
                
            # --- Familia 3: Failed Breakout Fade ---
            # brk_fail flag + Rejection flag
            if row['brk_fail'] == 1 or row['rejection_flag'] == 1:
                signals[i] = 4 # CLOSE
                
            # --- Familia 4: Range Mean Reversion ---
            # Regime 4 (Range) + Extreme Channel Position
            if row.get('htf_regime') == 4:
                if row['chan_pos'] < 0.1: signals[i] = 1 # Buy low
                elif row['chan_pos'] > 0.9: signals[i] = 4 # Sell high
                
        return signals

def evaluate_baseline(df, env_class):
    strategy = DeterministicBaseline(df)
    signals = strategy.generate_signals()
    
    env = env_class(df)
    obs, _ = env.reset()
    
    total_reward = 0
    for i in range(len(signals)):
        action = int(signals[i])
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done: break
        
    return env.equity, env.trades_count if hasattr(env, 'trades_count') else 0
