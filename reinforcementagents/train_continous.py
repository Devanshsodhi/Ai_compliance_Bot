# train_continuous.py
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from r1_agent.env import DocumentComplianceEnvContinuous

def train(save_path="models/continuous_agent.zip", total_timesteps=100_000, seed=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = DummyVecEnv([lambda: DocumentComplianceEnvContinuous(seed=seed)])
    model = SAC("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    train()
