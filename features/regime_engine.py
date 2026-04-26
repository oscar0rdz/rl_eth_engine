import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

def classify_regime(df):
    """
    Robust classification for TREND_UP, TREND_DOWN, CHOP, RANGE, BREAKOUT_FAIL
    """
    df = df.copy()
    
    # Pre-calculate components
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    slope_20 = (ema_20 - ema_20.shift(5)) / ema_20.shift(5)
    
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    atr_pct = atr / df['close']
    atr_percentile = atr_pct.rolling(100).rank(pct=True)
    
    regime = []
    for i in range(len(df)):
        s20 = slope_20.iloc[i]
        ap = atr_percentile.iloc[i]
        
        if ap > 0.90: # High Volatility Shock
            regime.append(5) # VOL_SHOCK / BREAKOUT_EXPANSION
        elif abs(s20) < 0.0003 and ap < 0.4:
            regime.append(3) # CHOP
        elif s20 > 0.0005:
            regime.append(1) # TREND_UP
        elif s20 < -0.0005:
            regime.append(2) # TREND_DOWN
        else:
            regime.append(4) # RANGE_CHANNEL
            
    return regime

def classify_volatility_state(df):
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    atr_pct = atr / df['close']
    return atr_pct.rolling(100).rank(pct=True)

def classify_breakout_context(df):
    """
    Detects if we are in a breakout failure context
    """
    df = df.copy()
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    
    # Breakout then close inside
    breakout_fail = ((df['high'] > high_20.shift(1)) & (df['close'] < high_20.shift(1))).astype(int)
    return breakout_fail
