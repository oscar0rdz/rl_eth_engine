import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
from rl_eth_engine.risk_manager import RiskManager

class ETHTradingEnvV4(gym.Env):
    """
    V4 Industrial Grade Environment
    Localized episodes (96-288 bars), Staged curriculum, Frozen execution rules.
    Actions Phase A: 0=SKIP, 1=ENTER_SMALL, 2=ENTER_FULL
    Actions Phase B: 0=SKIP, 1=ENTER_S, 2=ENTER_F, 3=HOLD, 4=REDUCE_50, 5=CLOSE
    """
    def __init__(self, df, training_phase=1, config_path=None):
        super().__init__()
        self.df = df
        self.training_phase = training_phase # 1: Phase A (Entry), 2: Phase B (Mgmt)
        
        if config_path is None:
            # Dynamic path resolution to find configs inside the project root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(base_dir, 'configs', 'reward_config.yaml')
            
        self.config = self._load_config(config_path)
        
        # Action space always 6, but sanitized in step()
        self.action_space = spaces.Discrete(6)
        # State: feature_cols (8 families) + 4 familia G features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns) + 4,), dtype=np.float32)
        
        self.df_values = df.values.astype(np.float32)
        self.df_columns = list(df.columns)
        self.close_idx = self.df_columns.index('close')
        self.atr_idx = self.df_columns.index('atr_ratio')
        
        self.initial_capital = self.config['env_params']['initial_capital']
        self.risk_manager = RiskManager()
        
        self.reset()

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Localized Episodic Structure: 96 to 288 velas (8h to 24h at 5m)
        episode_len = self.np_random.integers(96, 289)
        if len(self.df) > episode_len:
            self.start_idx = self.np_random.integers(0, len(self.df) - episode_len)
            self.end_idx = self.start_idx + episode_len
        else:
            self.start_idx, self.end_idx = 0, len(self.df) - 1
            
        self.current_step = self.start_idx
        self.capital = self.initial_capital
        self.position = 0.0
        self.avg_price = 0.0
        self.equity = self.initial_capital
        self.max_equity = self.initial_capital
        self.prev_unrealized_pnl = 0.0
        
        self.trade_history = [] 
        self.last_trade_pnl = 0.0
        self.last_trade_regime = 0
        self.exposure_steps = 0
        
        self.risk_manager.reset_daily(self.equity)
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df_values[self.current_step]
        price = row[self.close_idx]
        
        # familia G: Trade State (normalized)
        pos_side = 1.0 if self.position > 0 else 0.0
        pos_size_pct = (self.position * price) / self.equity
        unrealized = self.position * (price - self.avg_price) if self.position > 0 else 0.0
        unreal_pnl_pct = unrealized / self.equity
        bars_in_trade = self.exposure_steps / 288.0
        
        trade_state = np.array([pos_side, pos_size_pct, unreal_pnl_pct, bars_in_trade], dtype=np.float32)
        return np.concatenate([row, trade_state])

    def step(self, action):
        # FROZEN EXECUTION: Decision at t (current_idx), Fill at t+1 (next_idx)
        current_price = self.df.iloc[self.current_step]['close']
        next_idx = min(self.current_step + 1, len(self.df) - 1)
        next_price = self.df.iloc[next_idx]['close']
        
        prev_equity = self.equity
        
        # 1. Santize and Apply Action
        sanitized_action = self._sanitize_action(action)
        executed, penalty_type = self._apply_action(sanitized_action, next_price)
        
        # 2. Update Position State based on Next Price
        self.equity = self.capital + (self.position * next_price)
        new_unrealized = self.position * (next_price - self.avg_price) if self.position > 0 else 0.0
        unreal_delta = new_unrealized - self.prev_unrealized_pnl
        
        # 3. Phase A Auto-Exits
        if self.training_phase == 1 and self.position > 0:
            self.exposure_steps += 1
            if self._check_semi_exit(new_unrealized / self.equity, self.exposure_steps):
                self._sell(self.position, next_price, self.config['env_params']['trading_fee'])
        elif self.position > 0:
            self.exposure_steps += 1
        else:
            self.exposure_steps = 0

        # 4. Advance
        self.risk_manager.step()
        self.current_step = next_step = next_idx
        self.prev_unrealized_pnl = new_unrealized
        done = self.current_step >= self.end_idx
        
        # 4b. FROZEN RULE: End Flat
        if done and self.position > 0:
            self._sell(self.position, next_price, self.config['env_params']['trading_fee'])
        
        # 5. Reward V4
        reward = self._calculate_reward_v4(self.equity - prev_equity, unreal_delta, penalty_type)
        
        info = {
            'equity': self.equity,
            'trade_count': len(self.trade_history),
            'last_trade_pnl': self.last_trade_pnl,
            'last_trade_regime': self.last_trade_regime,
            'action': action,
            'sanitized_action': sanitized_action
        }
        self.last_trade_pnl = 0.0 # Reset
        
        return self._get_observation(), float(reward), done, False, info

    def _sanitize_action(self, action):
        # Phase A (Entry Only): Actions allowed are 0, 1, 2. If > 2, force SKIP (0) or HOLD (3)
        if self.training_phase == 1:
            if action > 2: return 0 
        # State awareness: If flat, actions 3,4,5 are HOLD or SKIP
        if self.position == 0 and action in [3,4,5]: return 0
        # If position open, actions 1,2 are HOLD
        if self.position > 0 and action in [1,2]: return 3
        return action

    def _apply_action(self, action, next_price):
        cfg = self.config['env_params']
        if action == 0: return True, None # SKIP
        if action == 3: return True, None # HOLD
        
        if action in [1, 2]: # ENTER
            can_open, _ = self.risk_manager.can_open(self.equity, cfg['spread'], self.df.iloc[self.current_step]['atr_ratio'])
            if not can_open: return False, "invalid"
            size = 0.25 if action == 1 else 0.50
            self._buy(self.equity * size, next_price * (1 + cfg['slippage']), cfg['trading_fee'])
            return True, "trade"
            
        if action == 4: # REDUCE_50
            self._sell(self.position * 0.5, next_price * (1 - cfg['slippage']), cfg['trading_fee'])
            return True, "trade"
            
        if action == 5: # CLOSE
            self._sell(self.position, next_price * (1 - cfg['slippage']), cfg['trading_fee'])
            return True, "trade"
            
        return False, None

    def _check_semi_exit(self, pnl_pct, steps):
        if pnl_pct < -0.01: return True # Stop Approx
        if pnl_pct > 0.015: return True # Take Approx
        if steps > 96: return True # Time stop
        return False

    def _buy(self, amount, price, fee):
        cost = amount * fee
        qty = (amount - cost) / price
        # Frozen size rounding (Approx to match paper)
        qty = np.floor(qty * 1000) / 1000.0 
        if qty * price < 10.0: return # min_notional
        
        if (self.position + qty) > 0:
            self.avg_price = (self.position * self.avg_price + qty * price) / (self.position + qty)
        self.position += qty
        self.capital -= (qty * price) + cost

    def _sell(self, qty, price, fee):
        rev = qty * price
        cost = rev * fee
        p_realized = (price - self.avg_price) * qty
        self.capital += (rev - cost)
        self.position -= qty
        
        # Log trade
        regime = self.df.iloc[self.current_step].get('htf_regime', 0)
        self.trade_history.append({'pnl': p_realized, 'regime': regime, 'bars': self.exposure_steps})
        self.last_trade_pnl = p_realized
        self.last_trade_regime = regime
        
        if self.position <= 0.00001:
            self.position, self.avg_price = 0, 0
            self.risk_manager.update_after_trade(p_realized)

    def _calculate_reward_v4(self, realized_delta, unreal_delta, penalty_type):
        norm = self.initial_capital
        # Reward = Realized + 0.15*UnrealDelta - Costs - 1.25*DD_Delta
        # Fees/Slippage are already in realized_delta since they reduce capital
        reward = (realized_delta / norm) * 1.0
        reward += (unreal_delta / norm) * 0.12 # User suggested 0.10-0.15
        
        # Drawdown delta
        dd_curr = max(0, (self.max_equity - self.equity) / self.max_equity)
        self.max_equity = max(self.max_equity, self.equity)
        # We penalize the INCREASE in DD
        reward -= dd_curr * 1.25
        
        if penalty_type == "invalid": reward -= 0.01
        # Stale position check
        if self.position > 0 and unreal_delta < 0 and self.exposure_steps > 48:
            reward -= 0.02
            
        return reward
