

import numpy as np
from collections import defaultdict
import pickle
import os

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
                 exploration_decay=0.9995, min_exploration=0.01):
       
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: np.zeros(2))
        
        # Training history
        self.training_history = {
            'episodes': [],
            'win_rates': [],
            'exploration_rates': [],
            'q_table_sizes': []
        }
    
    def choose_action(self, state):
        state_key = self._state_to_key(state)
        
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        
        # Exploitation: best action from Q-table
        else:
            return np.argmax(self.q_table[state_key])
    
    def step(self, state):
        return self.choose_action(state)
    
    def eval_step(self, state):
        state_key = self._state_to_key(state)
        action = np.argmax(self.q_table[state_key])
        return action, {}  # Return action and empty info dict
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning formula
        Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # Update Q-value
        self.q_table[state_key][action] += self.lr * (target - current_q)
        
        # Decay exploration rate
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _state_to_key(self, state):
        """Convert environment state to Q-table key (player_value, dealer_showing)"""
        # Robust handling for multiple possible state shapes:
        # - dict with 'player_value' and 'dealer_showing' (legacy expectation)
        # - dict with 'obs' (e.g. {'obs': [player_sum, dealer_showing]})
        # - dict with 'raw_obs' (RLCard raw_obs); try to decode via mc.decode_rlcard_state
        # - tuple/list like (player_value, dealer_showing)
        # Fallback to (0, 0) when we cannot extract values.
        try:
            # Direct keys (old environment)
            if isinstance(state, dict) and 'player_value' in state and 'dealer_showing' in state:
                return (int(state['player_value']), int(state['dealer_showing']))

            # Numeric observation array (common in RLCard wrappers)
            if isinstance(state, dict) and 'obs' in state:
                obs = state['obs']
                try:
                    return (int(obs[0]), int(obs[1]))
                except Exception:
                    pass

            # Raw RLCard observation: try to decode using mc.decode_rlcard_state if available
            if isinstance(state, dict) and 'raw_obs' in state and isinstance(state['raw_obs'], dict):
                try:
                    # Import lazily to avoid hard dependency if mc isn't present
                    from mc import decode_rlcard_state
                    pv, ds, _ = decode_rlcard_state(state['raw_obs'])
                    return (int(pv), int(ds))
                except Exception:
                    # Best-effort: try to extract numeric dealer showing from raw strings
                    ro = state['raw_obs']
                    try:
                        # player hand may be under 'player0 hand' or 'hand'
                        ph = ro.get('player0 hand') or ro.get('hand')
                        dh = ro.get('dealer hand')
                        # If it's impossible to decode, fallthrough to fallback below
                        # We could compute a naive sum, but prefer a safe default
                        pass
                    except Exception:
                        pass

            # If state itself is a pair/sequence
            if isinstance(state, (list, tuple)) and len(state) >= 2:
                try:
                    return (int(state[0]), int(state[1]))
                except Exception:
                    pass

        except Exception:
            # Fall through to default
            pass

        # Fallback: return a neutral key so Q-table lookups don't KeyError
        return (0, 0)
    
    def save_model(self, filename='models/q_learning_agent.pkl'):
        os.makedirs('models', exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"✓ Q-learning model saved to {filename}")
    
    def load_model(self, filename='models/q_learning_agent.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                loaded_table = pickle.load(f)
                self.q_table.update(loaded_table)
            print(f"✓ Q-learning model loaded from {filename}")
            return True
        else:
            print(f"✗ No saved model found at {filename}")
            return False

def train_q_learning_with_env(environment, num_episodes=50000, evaluation_interval=1000):
    
    print("=== TRAINING Q-LEARNING AGENT ===")
    print(f"Training for {num_episodes} episodes...")
    print(f"Evaluating progress every {evaluation_interval} episodes...")
    
    agent = QLearningAgent()
    evaluation_interval = max(1, evaluation_interval)
    
    for episode in range(num_episodes):
        state, player_id = environment.reset()
        episode_reward = 0
        
        # Play one episode
        while not environment.is_over():
            current_player = environment.get_player_id()
            
            if current_player == 0:  # Agent's turn
                # Get state info for Q-learning
                state_info = environment.get_state_info(state)
                action = agent.choose_action(state_info)
                next_state, next_player_id = environment.step(action)
                
                # Get reward when game ends
                if environment.is_over():
                    reward = environment.get_payoffs()[0]
                    episode_reward = reward
                else:
                    reward = 0
                
                # Update Q-table with state info
                next_state_info = environment.get_state_info(next_state)
                agent.update(state_info, action, reward, next_state_info, environment.is_over())
                state = next_state
                
            else:  # Dealer's turn
                next_state, next_player_id = environment.step(0)
                state = next_state
        
        # Record training progress at intervals
        if (episode + 1) % evaluation_interval == 0 or (episode + 1) == num_episodes:
            recent_wins = 0
            eval_games = min(500, evaluation_interval)
            
            for _ in range(eval_games):
                eval_state, _ = environment.reset()
                
                while not environment.is_over():
                    current_player = environment.get_player_id()
                    if current_player == 0:
                        state_info = environment.get_state_info(eval_state)
                        state_key = agent._state_to_key(state_info)
                        action = np.argmax(agent.q_table[state_key])  # Greedy
                        next_state, _ = environment.step(action)
                    else:
                        next_state, _ = environment.step(0)
                    eval_state = next_state
                
                if environment.get_payoffs()[0] > 0:
                    recent_wins += 1
            
            win_rate = recent_wins / eval_games
            
            # Record progress
            agent.training_history['episodes'].append(episode + 1)
            agent.training_history['win_rates'].append(win_rate)
            agent.training_history['exploration_rates'].append(agent.epsilon)
            agent.training_history['q_table_sizes'].append(len(agent.q_table))
            
            print(f"Episode {episode + 1:5d}: Win Rate = {win_rate:.3f}, "
                  f"Epsilon = {agent.epsilon:.3f}, "
                  f"Q-table = {len(agent.q_table):3d} states")
    
    print("✓ Training completed!")
    return agent, agent.training_history

def evaluate_agent(agent, environment, num_games=10000):
    """Evaluate the trained agent"""
    print(f"📊 Evaluating agent on {num_games} games...")
    
    wins = 0
    losses = 0
    ties = 0
    
    for game in range(num_games):
        state, _ = environment.reset()
        
        while not environment.is_over():
            current_player = environment.get_player_id()
            
            if current_player == 0:
                state_info = environment.get_state_info(state)
                state_key = agent._state_to_key(state_info)
                action = np.argmax(agent.q_table[state_key])  # Greedy policy
                next_state, _ = environment.step(action)
            else:
                next_state, _ = environment.step(0)
                
            state = next_state
        
        reward = environment.get_payoffs()[0]
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            ties += 1
    
    win_rate = wins / num_games
    print(f"Results: Wins={wins} ({win_rate:.1%}), Losses={losses}, Ties={ties}")
    return win_rate

# Test the Q-learning agent with your environment
if __name__ == "__main__":
    from fix_env import BlackjackEnvironment
    
    # Test that the agent works with your environment
    env = BlackjackEnvironment()
    agent = QLearningAgent()
    
    print("✓ Q-learning agent initialized successfully")
    print(f"Initial Q-table size: {len(agent.q_table)} states")
    
    # Test action selection
    test_state, _ = env.reset()
    test_state_info = env.get_state_info(test_state)
    action = agent.choose_action(test_state_info)
    print(f"Test action: {'HIT' if action == 1 else 'STAND'}")
    print(f"Test state: Player={test_state_info['player_value']}, Dealer={test_state_info['dealer_showing']}")