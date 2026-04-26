import os
import yaml
from sb3_contrib import RecurrentPPO
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnv

class CurriculumTrainer:
    def __init__(self, env, log_dir="./tensorboard_logs/curriculum"):
        self.env = env
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def train_stage_a(self, total_timesteps=300000):
        """Etapa A: Aprender a no morir (Acciones básicas, Riesgo bajo)"""
        print("Starting Etapa A: Survival training...")
        # We can pass specific config or wrap env if needed
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            self.env, 
            verbose=1,
            tensorboard_log=self.log_dir,
            n_steps=1024,
            batch_size=128,
            learning_rate=0.0003
        )
        model.learn(total_timesteps=total_timesteps, tb_log_name="Etapa_A")
        return model

    def train_stage_b(self, model, total_timesteps=1000000):
        """Etapa B: Aprender entradas (Acciones completas, Reward suavizado)"""
        print("Starting Etapa B: Entry/Exit optimization...")
        model.learn(total_timesteps=total_timesteps, tb_log_name="Etapa_B", reset_num_timesteps=False)
        return model

    def train_stage_c(self, model, total_timesteps=2000000):
        """Etapa C: Aprender robustez (Stress de costos, Walk-forward)"""
        print("Starting Etapa C: Robustness & Stress training...")
        # Here we could increase slippage/fees in the env before continuing
        self.env.config['env_params']['trading_fee'] *= 1.5
        model.learn(total_timesteps=total_timesteps, tb_log_name="Etapa_C", reset_num_timesteps=False)
        return model
