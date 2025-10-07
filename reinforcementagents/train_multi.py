# inference.py
import numpy as np
from stable_baselines3 import PPO, SAC
from r1_agent.env import doc_type_one_hot

def build_observation(model_pred, model_conf, missing_fields, doc_type_str, hist_success, n_missing_fields_max=10):
    missing_norm = min(missing_fields, n_missing_fields_max) / n_missing_fields_max
    doc_type_idx = 0
    try:
        doc_type_idx = ["invoice","po","grn"].index(doc_type_str)
    except ValueError:
        doc_type_idx = 0
    obs = np.concatenate([
        np.array([float(model_pred)], dtype=np.float32),
        np.array([float(model_conf)], dtype=np.float32),
        np.array([missing_norm], dtype=np.float32),
        doc_type_one_hot(doc_type_idx),
        np.array([float(hist_success)], dtype=np.float32)
    ])
    return obs

def inference_binary(model_path, obs):
    model = PPO.load(model_path)
    action, _ = model.predict(obs, deterministic=True)
    decision = "human_review" if int(action) == 0 else "pass"
    return decision, int(action)

def inference_continuous(model_path, obs, pass_threshold=0.7):
    model = SAC.load(model_path)
    action, _ = model.predict(obs, deterministic=True)
    conf = float(action[0])
    decision = "pass" if conf >= pass_threshold else "human_review"
    return decision, conf

def inference_multi(model_path, obs):
    model = PPO.load(model_path)
    action, _ = model.predict(obs, deterministic=True)
    mapping = {0: "approve", 1: "reject", 2: "request_more_info", 3: "human_review"}
    return mapping.get(int(action), "unknown"), int(action)

if __name__ == "__main__":
    # simple demo
    obs = build_observation(model_pred=1, model_conf=0.85, missing_fields=1, doc_type_str="invoice", hist_success=0.9)
    print("Obs shape:", obs.shape)
    # Replace with actual model paths
    # print(inference_binary("models/binary_agent.zip", obs))
    # print(inference_continuous("models/continuous_agent.zip", obs))
