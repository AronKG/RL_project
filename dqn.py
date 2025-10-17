import random
import collections
import time
import math
from typing import Any, Tuple
import numpy as np
import torch
import rlcard

"""
dqn.py

A minimal Deep Q-Network (DQN) implementation to interact with the RLCard 'blackjack' environment.

Notes:
- This file attempts to be robust to small API differences in rlcard environments:
    it inspects reset() and step() return values to extract an observation vector.
- You may need to adapt obs_to_vector() if your RLCard setup returns complex nested observations.
- Requires: rlcard, torch, numpy

Usage:
        python dqn.py

This will run a short training loop and print episode rewards. Adjust HYPERPARAMS below as needed.
"""


import torch.nn as nn
import torch.optim as optim



# -------------------------
# Hyperparameters
# -------------------------
ENV_NAME = "blackjack"
SEED = 42
NUM_EPISODES = 2000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
MEMORY_SIZE = 50000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000  # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000  # steps (exponential)
HIDDEN_SIZE = 128
MAX_STEPS_PER_EPISODE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Utilities: env wrappers
# -------------------------
def env_make(name: str):
        env = rlcard.make(name)
        env.set_seed(SEED)
        return env


def env_reset(env) -> Any:
        """Call env.reset() and return the raw state returned by the environment."""
        result = env.reset()
        # Some RLCard versions return just a state, others return (state, player_id)
        if isinstance(result, tuple) and len(result) == 2:
                return result[0]
        return result


def env_step(env, action: int) -> Tuple[Any, float, bool, dict]:
        """Call env.step(action) and normalize common return signatures to (next_state, reward, done, info)."""
        step_result = env.step(action)

        # Common gym-like: (next_state, reward, done, info)
        if isinstance(step_result, tuple):
                if len(step_result) == 4:
                        return step_result  # (next_state, reward, done, info)
                # Some RLCard environments return (next_state, next_player, done, reward, info)
                if len(step_result) == 5:
                        next_state, _next_player, done, reward, info = step_result
                        return next_state, reward, done, info
                # Some return (state, reward, done)
                if len(step_result) == 3:
                        next_state, reward, done = step_result
                        return next_state, reward, done, {}
        # Fallback: try to interpret as (state, )
        raise RuntimeError(f"Unrecognized env.step return signature: {type(step_result)} / {step_result}")


def obs_to_vector(state: Any) -> np.ndarray:
        """
        Convert various RLCard observation/state formats into a 1D float32 numpy vector.

        Adapts to:
            - dicts with 'obs' or 'raw_obs' keys (common in rlcard)
            - numpy arrays
            - lists/tuples of numeric values

        If your environment uses a different representation, modify this function accordingly.
        """
        if isinstance(state, dict):
                if "obs" in state:
                        arr = state["obs"]
                        return np.asarray(arr, dtype=np.float32).ravel()
                if "raw_obs" in state:
                        arr = state["raw_obs"]
                        return np.asarray(arr, dtype=np.float32).ravel()
                # sometimes 'state' key is used
                if "state" in state:
                        arr = state["state"]
                        return np.asarray(arr, dtype=np.float32).ravel()
                # try converting values in dict
                try:
                        flat = []
                        for k in sorted(state.keys()):
                                v = state[k]
                                flat.append(np.asarray(v, dtype=np.float32).ravel())
                        if flat:
                                return np.concatenate(flat).astype(np.float32)
                except Exception:
                        pass

        if isinstance(state, np.ndarray):
                return state.astype(np.float32).ravel()
        if isinstance(state, (list, tuple)):
                try:
                        return np.asarray(state, dtype=np.float32).ravel()
                except Exception:
                        pass

        raise ValueError(f"Unrecognized state format for vectorization: {type(state)}")


# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
        def __init__(self, capacity: int):
                self.capacity = capacity
                self.buffer = collections.deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
                self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size: int):
                batch = random.sample(self.buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                return (
                        np.stack(states),
                        np.array(actions, dtype=np.int64),
                        np.array(rewards, dtype=np.float32),
                        np.stack(next_states),
                        np.array(dones, dtype=np.float32),
                )

        def __len__(self):
                return len(self.buffer)


# -------------------------
# Network
# -------------------------
class MLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_size: int = HIDDEN_SIZE):
                super().__init__()
                self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                )

        def forward(self, x):
                return self.net(x)


# -------------------------
# DQN Agent
# -------------------------
class DQNAgent:
        def __init__(self, state_dim: int, action_dim: int):
                self.action_dim = action_dim

                self.policy_net = MLP(state_dim, action_dim).to(DEVICE)
                self.target_net = MLP(state_dim, action_dim).to(DEVICE)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()

                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
                self.replay = ReplayBuffer(MEMORY_SIZE)

                self.steps_done = 0

        def select_action(self, state_vec: np.ndarray, epsilon: float) -> int:
                # epsilon-greedy
                if random.random() < epsilon:
                        return random.randrange(self.action_dim)
                state_t = torch.from_numpy(state_vec).float().to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                        qvals = self.policy_net(state_t)
                return int(qvals.argmax(dim=1).item())

        def push_transition(self, s, a, r, ns, done):
                self.replay.push(s, a, r, ns, done)

        def optimize(self):
                if len(self.replay) < BATCH_SIZE:
                        return None

                states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)
                states_t = torch.from_numpy(states).float().to(DEVICE)
                actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(DEVICE)
                rewards_t = torch.from_numpy(rewards).float().unsqueeze(1).to(DEVICE)
                next_states_t = torch.from_numpy(next_states).float().to(DEVICE)
                dones_t = torch.from_numpy(dones).float().unsqueeze(1).to(DEVICE)

                # Current Q values
                q_values = self.policy_net(states_t).gather(1, actions_t)

                # Compute target Q values
                with torch.no_grad():
                        next_q_values = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
                        target_q_values = rewards_t + (1.0 - dones_t) * GAMMA * next_q_values

                loss = nn.functional.mse_loss(q_values, target_q_values)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
                self.optimizer.step()

                return loss.item()

        def update_target(self):
                self.target_net.load_state_dict(self.policy_net.state_dict())

        def save(self, path: str):
                torch.save(self.policy_net.state_dict(), path)

        def load(self, path: str):
                self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
                self.update_target()


# -------------------------
# Training loop
# -------------------------
def linear_epsilon(step: int):
        # exponential decay
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)
        return eps


def main():
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        env = env_make(ENV_NAME)

        # Build a sample state vector to infer dimensions
        raw_state = env_reset(env)
        state_vec = obs_to_vector(raw_state)
        state_dim = state_vec.shape[0]

        # Determine number of actions
        if hasattr(env, "action_num"):
                action_dim = env.action_num
        elif hasattr(env, "action_space"):
                try:
                        action_dim = env.action_space.n
                except Exception:
                        action_dim = getattr(env.action_space, "shape", ())[0]
        elif hasattr(env, "num_actions"):
                action_dim = env.num_actions
        else:
                # Try taking a random action returned by env.available_actions if exists
                if hasattr(env, "action_list"):
                        action_dim = len(env.action_list)
                else:
                        raise RuntimeError("Cannot infer action dimension from environment; inspect env object.")

        print(f"Environment: {ENV_NAME}, state_dim={state_dim}, action_dim={action_dim}, device={DEVICE}")

        agent = DQNAgent(state_dim, action_dim)

        total_steps = 0
        episode_rewards = []

        # Pre-fill replay with random policy
        print("Populating replay buffer with random transitions...")
        while len(agent.replay) < MIN_REPLAY_SIZE:
                state = env_reset(env)
                s_vec = obs_to_vector(state)
                done = False
                t = 0
                while not done and t < MAX_STEPS_PER_EPISODE:
                        action = random.randrange(action_dim)
                        next_state, reward, done, _ = env_step(env, action)
                        ns_vec = obs_to_vector(next_state)
                        agent.push_transition(s_vec, action, reward, ns_vec, float(done))
                        s_vec = ns_vec
                        t += 1

        print(f"Replay buffer size: {len(agent.replay)}")

        start_time = time.time()
        for ep in range(1, NUM_EPISODES + 1):
                state = env_reset(env)
                s_vec = obs_to_vector(state)
                done = False
                ep_reward = 0.0
                t = 0
                while not done and t < MAX_STEPS_PER_EPISODE:
                        epsilon = linear_epsilon(total_steps)
                        action = agent.select_action(s_vec, epsilon)
                        next_state, reward, done, _ = env_step(env, action)
                        ns_vec = obs_to_vector(next_state)

                        agent.push_transition(s_vec, action, reward, ns_vec, float(done))
                        loss = agent.optimize()

                        s_vec = ns_vec
                        ep_reward += reward
                        t += 1
                        total_steps += 1

                        # Update target network
                        if total_steps % TARGET_UPDATE_FREQ == 0:
                                agent.update_target()

                episode_rewards.append(ep_reward)

                if ep % 50 == 0 or ep == 1:
                        avg_r = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 1 else float(ep_reward)
                        print(f"Episode {ep}/{NUM_EPISODES} | steps {total_steps} | recent_avg_reward {avg_r:.3f} | epsilon {epsilon:.3f}")

        duration = time.time() - start_time
        print(f"Training completed in {duration:.1f}s. Saving policy to dqn_policy.pth")
        agent.save("dqn_policy.pth")


if __name__ == "__main__":
        main()