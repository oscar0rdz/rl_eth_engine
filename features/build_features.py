import os
import pandas as pd
import numpy as np
import yaml
from binance.client import Client
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from rl_eth_engine.features.regime_engine import classify_regime, classify_volatility_state, classify_breakout_context
import time



def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_binance_data(symbol, interval, start_str, end_str=None):
    client = Client()
    print(f"Fetching {symbol} {interval} from {start_str}...")
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df[numeric_cols]

def build_features(df_5m, df_15m, df_1h):
    """
    V3 Industrial Grade Feature Engineering - 8 Families
    """
    df = df_5m.copy()
    epsilon = 1e-9
    
    # --- Familia A: Geometría ---
    df['body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + epsilon)
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + epsilon)
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + epsilon)
    df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'] + epsilon)
    df['range_expansion'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(20).mean()
    df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(float)
    df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(float)
    
    # --- Familia B: Tendencia ---
    ema20 = EMAIndicator(df['close'], window=20).ema_indicator()
    ema50 = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema20_slope'] = ema20.pct_change(5)
    df['ema50_slope'] = ema50.pct_change(5)
    df['ema20_above_50'] = (ema20 > ema50).astype(float)
    df['dist_ema20'] = (df['close'] - ema20) / (ema20 + epsilon)
    df['dist_ema50'] = (df['close'] - ema50) / (ema50 + epsilon)
    
    # Rolling regression slope (proxy)
    df['reg_slope'] = df['close'].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x[0], raw=True)
    df['trend_strength'] = ADXIndicator(df['high'], df['low'], df['close']).adx() / 100.0
    
    # --- Familia C: Valor ---
    vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['dist_vwap'] = (df['close'] - vwap) / (vwap + epsilon)
    df['zscore_vwap'] = (df['close'] - vwap) / (df['close'].rolling(20).std() + epsilon)
    
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    mid_20 = (high_20 + low_20) / 2
    df['dist_chan_mid'] = (df['close'] - mid_20) / (mid_20 + epsilon)
    df['chan_pos'] = (df['close'] - low_20) / (high_20 - low_20 + epsilon)
    
    # --- Familia D: Volatilidad ---
    df['atr_ratio'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range() / (df['close'] + epsilon)
    df['true_range'] = (df['high'] - df['low']) / (df['close'] + epsilon)
    df['realized_vol_12'] = df['close'].pct_change().rolling(12).std()
    df['realized_vol_24'] = df['close'].pct_change().rolling(24).std()
    
    # --- Familia E: Volumen ---
    df['vol_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + epsilon)
    vol_up = df['volume'].where(df['close'] > df['open'], 0).rolling(20).sum()
    vol_down = df['volume'].where(df['close'] < df['open'], 0).rolling(20).sum()
    df['vol_up_ratio'] = vol_up / (vol_up + vol_down + epsilon)
    df['vol_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    
    # --- Familia F: Contexto HTF (1h) ---
    ema_1h = EMAIndicator(df_1h['close'], window=20).ema_indicator()
    vwap_1h = VolumeWeightedAveragePrice(df_1h['high'], df_1h['low'], df_1h['close'], df_1h['volume']).volume_weighted_average_price()
    
    df['htf_trend_up'] = (ema_1h.pct_change(5) > 0.0001).astype(float).reindex(df.index, method='ffill')
    df['htf_trend_down'] = (ema_1h.pct_change(5) < -0.0001).astype(float).reindex(df.index, method='ffill')
    df['htf_dist_vwap'] = ((df_1h['close'] - vwap_1h) / vwap_1h).reindex(df.index, method='ffill')
    
    if 'classify_regime' in globals():
        df['htf_regime'] = pd.Series(classify_regime(df_1h), index=df_1h.index).reindex(df.index, method='ffill')
    else: df['htf_regime'] = 0
    
    # --- Familia H: Proxies de Trampa ---
    df['brk_fail'] = classify_breakout_context(df)
    df['rejection_flag'] = ((df['upper_wick_pct'] > 0.6) & (df['vol_zscore'] > 1.0)).astype(float)
    df['vol_spike_reject'] = ((df['volume'] > df['volume'].rolling(50).mean() * 2) & (df['body_pct'] < 0.3)).astype(float)
    
    # Cleanup
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # Familia G: Estado del Trade (Placeholders)
    # These are strictly updated in the environment
    state_cols = [
        'position_side', 'position_size_pct', 'unrealized_pnl_pct', 
        'bars_in_trade', 'mfe_pct', 'mae_pct', 'dist_to_stop', 'daily_drawdown'
    ]
    
    feature_cols = [
        'close', # Internal use
        'body_pct', 'upper_wick_pct', 'lower_wick_pct', 'close_location', 'range_expansion', 'inside_bar', 'outside_bar', # A
        'ema20_slope', 'ema50_slope', 'ema20_above_50', 'dist_ema20', 'dist_ema50', 'reg_slope', 'trend_strength', # B
        'dist_vwap', 'zscore_vwap', 'dist_chan_mid', 'chan_pos', # C
        'atr_ratio', 'true_range', 'realized_vol_12', 'realized_vol_24', # D
        'vol_zscore', 'vol_up_ratio', 'vol_trend', # E
        'htf_trend_up', 'htf_trend_down', 'htf_dist_vwap', 'htf_regime', # F
        'brk_fail', 'rejection_flag', 'vol_spike_reject' # H
    ]
    
    for sc in state_cols:
        df[sc] = 0.0
        feature_cols.append(sc)
        
    return df[feature_cols]




if __name__ == "__main__":
    # Example usage (would be called with real or saved data)
    pass
