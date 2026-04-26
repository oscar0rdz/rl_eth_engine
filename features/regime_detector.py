import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

def detect_regime(df):
    """
    Classifies market regime based on price action and volatility.
    Regimes: TREND_UP, TREND_DOWN, CHOP, RANGE_CHANNEL, BREAKOUT_EXPANSION, LOW_LIQUIDITY
    """
    df = df.copy()
    
    # 1. Trend (EMA Slope)
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    slope = (ema_20 - ema_20.shift(5)) / ema_20.shift(5)
    
    # 2. Volatility (ATR Percentile)
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    atr_pct = atr / df['close']
    atr_percentile = atr_pct.rolling(window=100).rank(pct=True)
    
    # 3. Decision Logic
    regime = []
    for i in range(len(df)):
        if atr_percentile.iloc[i] < 0.1:
            regime.append("LOW_LIQUIDITY")
        elif abs(slope.iloc[i]) < 0.0005 and atr_percentile.iloc[i] < 0.4:
            regime.append("CHOP")
        elif slope.iloc[i] > 0.001 and atr_percentile.iloc[i] > 0.6:
            regime.append("BREAKOUT_EXPANSION")
        elif slope.iloc[i] > 0.0005:
            regime.append("TREND_UP")
        elif slope.iloc[i] < -0.0005:
            regime.append("TREND_DOWN")
        else:
            regime.append("RANGE_CHANNEL")
            
    # Map to IDs
    regime_map = {
        "TREND_UP": 1,
        "TREND_DOWN": 2,
        "CHOP": 3,
        "RANGE_CHANNEL": 4,
        "BREAKOUT_EXPANSION": 5,
        "MANIPULATION_SPIKE": 6,
        "LOW_LIQUIDITY": 7
    }
    
    return [regime_map.get(r, 4) for r in regime]

def get_market_regime_features(df):
    df = df.copy()
    df['ema_slope_1h'] = df['close'].ewm(span=12).mean().pct_change(12) # ~1h
    df['ema_slope_15m'] = df['close'].ewm(span=3).mean().pct_change(3) # ~15m
    df['atr_percentile'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range() / df['close']
    # Normalize
    df['atr_percentile'] = df['atr_percentile'].rolling(100).rank(pct=True)
    return df[['ema_slope_1h', 'ema_slope_15m', 'atr_percentile']]
