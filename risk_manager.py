class RiskManager:
    """
    Institutional Risk Manager (Capa 3)
    """
    def __init__(self, daily_dd_limit=0.015, spread_limit=0.001, cooldown_steps=24):
        self.daily_dd_limit = daily_dd_limit
        self.spread_limit = spread_limit
        self.cooldown_steps = cooldown_steps
        
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.daily_start_equity = 1000.0

    def can_open(self, current_equity, current_spread, current_atr_ratio):
        if self.cooldown_remaining > 0:
            return False, "Cooldown Active"
            
        daily_dd = (self.daily_start_equity - current_equity) / self.daily_start_equity
        if daily_dd > self.daily_dd_limit:
            return False, "Daily DD Limit Hit"
            
        if current_spread > self.spread_limit:
            return False, "Spread Too High"
            
        return True, "OK"

    def update_after_trade(self, profit):
        if profit < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                self.cooldown_remaining = self.cooldown_steps
        else:
            self.consecutive_losses = 0

    def step(self):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def reset_daily(self, equity):
        """Called at the start of each episode (day)"""
        self.daily_start_equity = equity
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
