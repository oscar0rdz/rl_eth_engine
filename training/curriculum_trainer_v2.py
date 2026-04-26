import os
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_eth_engine.envs.eth_trading_env import ETHTradingEnv

class CurriculumTrainerV2:
    def __init__(self, df, log_dir="./tensorboard_logs/v2_curriculum"):
        self.df = df
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def _create_env(self, friction_multiplier=1.0):
        # Create env and override friction
        env = ETHTradingEnv(self.df)
        env.config['env_params']['trading_fee'] *= friction_multiplier
        env.config['env_params']['slippage'] *= friction_multiplier
        
        env = Monitor(env)
        venv = DummyVecEnv([lambda: env])
        # Wrap with VecNormalize (will save stats later)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        return venv

    def run_stage(self, model=None, stage_name="Etapa_A", friction=1.0, steps=300000):
        print(f"--- Starting {stage_name} (Friction: {friction}x) ---")
        venv = self._create_env(friction)
        
        if model is None:
            model = RecurrentPPO(
                "MlpLstmPolicy", 
                venv, 
                verbose=1,
                tensorboard_log=self.log_dir,
                n_steps=1024,
                batch_size=128,
                learning_rate=0.0003
            )
        else:
            model.set_env(venv)
            
        model.learn(total_timesteps=steps, tb_log_name=stage_name, reset_num_timesteps=False)
        
        # Save model and stats
        model_path = os.path.join(self.log_dir, f"model_{stage_name}")
        model.save(model_path)
        venv.save(os.path.join(self.log_dir, f"stats_{stage_name}.pkl"))
        
        return model, venv
