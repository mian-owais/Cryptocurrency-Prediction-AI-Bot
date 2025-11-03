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
from typing import Any, Optional
import os
import numpy as np

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
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss="mse")
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
            raise RuntimeError("Keras/TensorFlow is required to train the DQN agent")

        # Simple replay buffer
        memory = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.act(np.array(state), epsilon=max(0.01, 0.1 * (1 - ep / episodes)))
                next_state, reward, done, info = env.step(action)
                memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(memory) >= batch_size:
                    batch = np.random.choice(len(memory), batch_size, replace=False)
                    states = np.array([memory[i][0] for i in batch])
                    actions = np.array([memory[i][1] for i in batch])
                    rewards = np.array([memory[i][2] for i in batch])
                    next_states = np.array([memory[i][3] for i in batch])
                    dones = np.array([memory[i][4] for i in batch])

                    q_next = self.model.predict(next_states, verbose=0)
                    q_target = rewards + (1 - dones) * gamma * np.max(q_next, axis=1)
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
    agent = DQNAgent(state_size=5, action_size=3) if KERAS_AVAILABLE else DQNAgent(5, 3)
    env = DummyEnv(5)
    try:
        agent.train(env, episodes=2)
        agent.save("models/rl_agent_demo.h5")
        print("Trained demo agent and saved to models/rl_agent_demo.h5")
    except Exception as e:
        print("Training demo skipped or failed:", e)
