"""rl_agent.py
Simple DQN agent for discrete actions: Predict Up / Down / Constant.

This is intentionally minimal and focused on local development and experimentation.
Requires: numpy, torch (optional). The agent will gracefully degrade if torch
is not available (it will still expose the same interface but training will be a no-op).
"""
from typing import Any, Optional
from collections import deque
import random
import os
from typing import Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, k=batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim)
            )

        def forward(self, x):
            return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: str = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

        self.device = device or (
            "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

        if TORCH_AVAILABLE:
            self.q_net = QNetwork(state_size, action_size).to(self.device)
            self.target_net = QNetwork(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
            self.update_count = 0
        else:
            # Dummy placeholders when torch isn't available
            self.q_net = None
            self.target_net = None
            self.optimizer = None

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection. state is a 1D numpy array."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        if TORCH_AVAILABLE:
            self.q_net.eval()
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
                q = self.q_net(s).cpu().numpy().squeeze(0)
            return int(np.argmax(q))
        else:
            # fallback: random
            return random.randrange(self.action_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if not TORCH_AVAILABLE:
            return  # no-op without torch

        if len(self.buffer) < max(128, self.batch_size):
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size)

        states_v = torch.tensor(
            states, dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(
            actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_v = torch.tensor(
            rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_v = torch.tensor(
            next_states, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones.astype(
            np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.q_net(states_v).gather(1, actions_v)
        next_q = self.target_net(next_states_v).max(1)[0].detach().unsqueeze(1)
        target = rewards_v + (1.0 - dones_v) * (self.gamma * next_q)

        loss = nn.functional.mse_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network occasionally
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if TORCH_AVAILABLE:
            torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        if TORCH_AVAILABLE and os.path.exists(path):
            self.q_net.load_state_dict(
                torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train_from_logs(self, verification_log_path: str, feature_keys: List[str]):
        """Train agent from past verification logs. Expects CSV with state columns matching feature_keys,
        action (0/1/2), reward, next_state columns and done flag. This is a convenience loader for periodic
        offline training (replay from historical transitions).
        """
        if not TORCH_AVAILABLE:
            print("Torch not available, skipping offline training")
            return

        import pandas as pd
        if not os.path.exists(verification_log_path):
            return

        df = pd.read_csv(verification_log_path)
        # Expect columns: feature_* for each feature in feature_keys, action, reward, next_feature_*
        for _, row in df.iterrows():
            try:
                s = np.array([row.get(f, 0.0)
                             for f in feature_keys], dtype=np.float32)
                a = int(row.get('action', 0))
                r = float(row.get('reward', 0.0))
                ns = np.array([row.get(f'next_{f}', row.get(
                    f, 0.0)) for f in feature_keys], dtype=np.float32)
                done = bool(row.get('done', False))
                self.store_transition(s, a, r, ns, done)
            except Exception:
                continue

        # perform multiple training passes
        for _ in range(200):
            self.train_step()


"""
rl_agent.py
-----------
Simple Deep Q-Learning (DQN) agent for trading demonstration.

This implementation tries to use TensorFlow/Keras if available; if not,
it provides a lightweight placeholder class with the same interface so the
rest of the system can import it during development.

API:
 - DQNAgent(state_size, action_size)
 - agent.train(env, episodes)
 - agent.save(path), agent.load(path)

Notes:
 - This is a demo-friendly implementation, not production-grade.
 - The environment expected is a simple custom env that yields state vectors.
"""

try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False


class DQNAgent:
    """A minimal DQN agent using Keras (if available)."""

    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.model = None
        if KERAS_AVAILABLE:
            self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(
            learning_rate=self.lr), loss="mse")
        self.model = model

    def act(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon or self.model is None:
            return int(np.random.randint(0, self.action_size))
        q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return int(np.argmax(q))

    def train(self, env: Any, episodes: int = 10, batch_size: int = 32, gamma: float = 0.99):
        """Train the agent on a simple env supporting reset() and step(action).

        This training loop is intentionally small for demo purposes.
        """
        if not KERAS_AVAILABLE:
            raise RuntimeError(
                "Keras/TensorFlow is required to train the DQN agent")

        # Simple replay buffer
        memory = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.act(np.array(state), epsilon=max(
                    0.01, 0.1 * (1 - ep / episodes)))
                next_state, reward, done, info = env.step(action)
                memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(memory) >= batch_size:
                    batch = np.random.choice(
                        len(memory), batch_size, replace=False)
                    states = np.array([memory[i][0] for i in batch])
                    actions = np.array([memory[i][1] for i in batch])
                    rewards = np.array([memory[i][2] for i in batch])
                    next_states = np.array([memory[i][3] for i in batch])
                    dones = np.array([memory[i][4] for i in batch])

                    q_next = self.model.predict(next_states, verbose=0)
                    q_target = rewards + (1 - dones) * \
                        gamma * np.max(q_next, axis=1)
                    q_vals = self.model.predict(states, verbose=0)
                    for i, a in enumerate(actions):
                        q_vals[i, a] = q_target[i]
                    self.model.train_on_batch(states, q_vals)

            # end episode
        return

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if KERAS_AVAILABLE and self.model is not None:
            self.model.save(path)
        else:
            # placeholder: save minimal metadata
            with open(path, "wb") as f:
                f.write(b"rl-agent-placeholder")

    def load(self, path: str) -> None:
        if KERAS_AVAILABLE:
            self.model = keras.models.load_model(path)
        else:
            # nothing to load in placeholder mode
            pass


class DummyEnv:
    """A tiny environment for testing the DQNAgent. State is random; reward is random.

    Replace with a proper trading environment for real training.
    """

    def __init__(self, state_size: int = 5):
        self.state_size = state_size
        self.reset()

    def reset(self):
        self.t = 0
        self.state = np.random.randn(self.state_size)
        return self.state

    def step(self, action: int):
        self.t += 1
        next_state = np.random.randn(self.state_size)
        reward = float(np.random.randn() * 0.01)
        done = self.t > 50
        info = {}
        return next_state, reward, done, info


if __name__ == "__main__":
    print("DQN agent module — demo mode")
    agent = DQNAgent(
        state_size=5, action_size=3) if KERAS_AVAILABLE else DQNAgent(5, 3)
    env = DummyEnv(5)
    try:
        agent.train(env, episodes=2)
        agent.save("models/rl_agent_demo.h5")
        print("Trained demo agent and saved to models/rl_agent_demo.h5")
    except Exception as e:
        print("Training demo skipped or failed:", e)
