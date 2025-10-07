# env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd

# Utility: one-hot for doc type
DOC_TYPES = ["invoice", "po", "grn"]

def doc_type_one_hot(doc_type_idx):
    vec = np.zeros(len(DOC_TYPES), dtype=np.float32)
    vec[doc_type_idx] = 1.0
    return vec

class DocumentComplianceEnv(gym.Env):
    """
    Discrete-action environment:
      - actions: 0 -> request human review, 1 -> pass
    Observation vector:
      [model_pred (0/1), model_confidence (0-1), missing_fields (0..N)/norm,
       doc_type_onehot..., historical_success_rate (0-1)]
    Reward:
      +1 correct decision, -2 for false positive (passing non-compliant),
      -0.5 for unnecessary human review (optional)
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 n_missing_fields_max=10,
                 seed: int = 0,
                 use_real_data=None):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        # observation: model_pred (1), confidence (1), missing_fields (1), doc_type one-hot (len(DOC_TYPES)), hist_success (1)
        obs_dim = 1 + 1 + 1 + len(DOC_TYPES) + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: human review, 1: pass
        self.n_missing_fields_max = n_missing_fields_max
        self.current = None
        self.use_real_data = use_real_data
        self.real_df = None
        if use_real_data is not None:
            self._load_real_data(use_real_data)

    def _load_real_data(self, csv_path):
        self.real_df = pd.read_csv(csv_path)
        # Expect columns: model_pred, model_conf, missing_fields, doc_type (str), hist_success, label (0/1)
        # Normalize missing_fields -> /n_missing_fields_max if larger
        self.real_df['missing_fields'] = self.real_df['missing_fields'].clip(0, self.n_missing_fields_max) / self.n_missing_fields_max
        # doc_type -> index
        self.real_df['doc_type_idx'] = self.real_df['doc_type'].apply(lambda x: DOC_TYPES.index(x) if x in DOC_TYPES else 0)

    def _sample_synthetic(self):
        # Sampling logic for synthetic training data
        # true label (compliant) probability depends on model_pred and missing_fields
        model_conf = self.rng.beta(2, 1)  # skew toward higher confidence
        model_pred = 1 if model_conf > 0.5 else 0
        missing_fields = self.rng.randint(0, self.n_missing_fields_max + 1)
        missing_norm = missing_fields / self.n_missing_fields_max
        doc_type_idx = self.rng.randint(0, len(DOC_TYPES))
        hist_success = self.rng.rand()

        # Ground truth: combine signals
        prob_compliant = 0.5 * model_conf + 0.3 * (1 - missing_norm) + 0.2 * hist_success
        true_label = 1 if self.rng.rand() < prob_compliant else 0

        obs = np.concatenate([
            np.array([float(model_pred)], dtype=np.float32),
            np.array([float(model_conf)], dtype=np.float32),
            np.array([missing_norm], dtype=np.float32),
            doc_type_one_hot(doc_type_idx),
            np.array([float(hist_success)], dtype=np.float32)
        ])
        return obs, true_label

    def _sample_from_df(self):
        idx = self.rng.randint(0, len(self.real_df))
        row = self.real_df.iloc[idx]
        obs = np.concatenate([
            np.array([float(row['model_pred'])], dtype=np.float32),
            np.array([float(row['model_conf'])], dtype=np.float32),
            np.array([float(row['missing_fields'])], dtype=np.float32),
            doc_type_one_hot(int(row['doc_type_idx'])),
            np.array([float(row['hist_success'])], dtype=np.float32)
        ])
        true_label = int(row['label'])
        return obs, true_label

    def reset(self):
        if self.real_df is not None:
            obs, label = self._sample_from_df()
        else:
            obs, label = self._sample_synthetic()
        self.current = {"obs": obs, "label": label}
        return obs

    def step(self, action):
        """
        action: 0 human_review (safe), 1 pass
        """
        obs = self.current["obs"]
        label = self.current["label"]
        done = True  # one-step episodic environment for training decision policy
        reward = 0.0

        if action == 0:
            # Human review: small negative cost (time/effort) but we assume final label will be correct
            # If human reviews, final decision equals true label -> reward depends on true label:
            if label == 1:
                # human reviewed and approves -> small positive
                reward = 0.5
            else:
                # human reviewed and rejects -> small positive (catching non-compliant)
                reward = 1.0
            info = {"outcome": "human_reviewed", "true_label": label}
        elif action == 1:
            # pass: if label==1 good; if label==0 big penalty
            if label == 1:
                reward = 1.0
            else:
                reward = -2.0  # severe penalty for false positive (passing non-compliant)
            info = {"outcome": "passed", "true_label": label}
        else:
            reward = -1.0
            info = {"outcome": "invalid_action", "true_label": label}

        # next state irrelevant; return zeros
        return obs, float(reward), done, info

class DocumentComplianceEnvContinuous(gym.Env):
    """
    Continuous-action environment where action is a confidence score in [0,1].
    Integration idea (for inference): if scored_confidence >= threshold -> pass else human review.
    Reward: encourage calibrated confidence:
       reward = 1 - abs(action - label)  (so closer to label yields higher reward)
       Penalize high confidence for wrong approvals strongly.
    """
    metadata = {"render.modes": []}

    def __init__(self, n_missing_fields_max=10, seed: int = 0, use_real_data=None):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        obs_dim = 1 + 1 + 1 + len(DOC_TYPES) + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        # action is scalar confidence in [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.n_missing_fields_max = n_missing_fields_max
        self.current = None
        self.use_real_data = use_real_data
        self.real_df = None
        if use_real_data is not None:
            self._load_real_data(use_real_data)

    def _load_real_data(self, csv_path):
        # same as discrete env
        import pandas as pd
        self.real_df = pd.read_csv(csv_path)
        self.real_df['missing_fields'] = self.real_df['missing_fields'].clip(0, self.n_missing_fields_max) / self.n_missing_fields_max
        self.real_df['doc_type_idx'] = self.real_df['doc_type'].apply(lambda x: DOC_TYPES.index(x) if x in DOC_TYPES else 0)

    def _sample_synthetic(self):
        model_conf = self.rng.beta(2, 1)
        model_pred = 1 if model_conf > 0.5 else 0
        missing_fields = self.rng.randint(0, self.n_missing_fields_max + 1)
        missing_norm = missing_fields / self.n_missing_fields_max
        doc_type_idx = self.rng.randint(0, len(DOC_TYPES))
        hist_success = self.rng.rand()
        prob_compliant = 0.5 * model_conf + 0.3 * (1 - missing_norm) + 0.2 * hist_success
        true_label = 1 if self.rng.rand() < prob_compliant else 0
        obs = np.concatenate([
            np.array([float(model_pred)], dtype=np.float32),
            np.array([float(model_conf)], dtype=np.float32),
            np.array([missing_norm], dtype=np.float32),
            doc_type_one_hot(doc_type_idx),
            np.array([float(hist_success)], dtype=np.float32)
        ])
        return obs, true_label

    def _sample_from_df(self):
        idx = self.rng.randint(0, len(self.real_df))
        row = self.real_df.iloc[idx]
        obs = np.concatenate([
            np.array([float(row['model_pred'])], dtype=np.float32),
            np.array([float(row['model_conf'])], dtype=np.float32),
            np.array([float(row['missing_fields'])], dtype=np.float32),
            doc_type_one_hot(int(row['doc_type_idx'])),
            np.array([float(row['hist_success'])], dtype=np.float32)
        ])
        true_label = int(row['label'])
        return obs, true_label

    def reset(self):
        if self.real_df is not None:
            obs, label = self._sample_from_df()
        else:
            obs, label = self._sample_synthetic()
        self.current = {"obs": obs, "label": label}
        return obs

    def step(self, action):
        # action is array-like [confidence]
        action = float(np.clip(action[0], 0.0, 1.0))
        obs = self.current["obs"]
        label = self.current["label"]
        done = True

        # reward: high if action close to label.
        # But penalize overconfident wrong approvals: if action > 0.5 and label==0 -> strong negative
        base_reward = 1.0 - abs(action - float(label))  # between 0 and 1
        penalty = 0.0
        if action > 0.7 and label == 0:
            penalty = -2.0
        reward = base_reward + penalty

        info = {"action_confidence": action, "true_label": label}
        return obs, float(reward), done, info
