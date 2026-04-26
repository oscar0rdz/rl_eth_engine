class RiskManager:
    def __init__(self, config):
        self.config = config
        self.daily_loss = 0.0
        self.max_daily_loss = 0.02 # 2%
        self.max_drawdown = 0.05   # 5%
        
    def validate_action(self, action, current_equity, initial_equity, current_drawdown):
        """
        Returns True if action is allowed by risk rules.
        """
        if current_drawdown > self.max_drawdown:
            print("Risk Manager: Max drawdown exceeded. Blocking action.")
            return False
            
        # Add more rules (e.g., daily loss)
        return True
