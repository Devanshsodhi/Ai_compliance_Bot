# train_binary.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from r1_agent.env import DocumentComplianceEnv

def train(save_path="models/binary_agent.zip", total_timesteps=50_000, seed=42):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = DummyVecEnv([lambda: DocumentComplianceEnv(seed=seed)])
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    train()
