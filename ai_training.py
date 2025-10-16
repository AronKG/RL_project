#!/usr/bin/env python3

import rlcard
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from tqdm import tqdm
import random

def decode_rlcard_state(raw_obs):
    """
    Decode RLCard raw observation to (player_sum, dealer_showing, usable_ace)
    """
    player_hand = raw_obs['player0 hand']
    dealer_hand = raw_obs['dealer hand']
    
    # Calculate player sum and usable ace
    player_sum, player_usable_ace = calculate_hand_value_and_ace(player_hand)
    
    # Get dealer's first visible card only
    if dealer_hand:
        dealer_card = dealer_hand[0]
        # Extract rank from RLCard format (e.g., 'D7' -> 7, 'CA' -> 1, 'CQ' -> 10)
        rank = dealer_card[-1]  # Last character is the rank
        if rank == 'A':
            dealer_showing = 1
        elif rank in ['J', 'Q', 'K', 'T']:  # T = 10
            dealer_showing = 10
        else:
            dealer_showing = int(rank)
    else:
        dealer_showing = 0  # Fallback if no dealer card
    
    return player_sum, dealer_showing, player_usable_ace

def calculate_hand_value_and_ace(hand):
    """
    Calculate hand value and check for usable ace
    Returns: (value, has_usable_ace)
    """
    # Extract ranks from RLCard format
    ranks = []
    for card in hand:
        rank = card[-1]  # Last character is the rank
        if rank == 'A':
            ranks.append(1)
        elif rank in ['J', 'Q', 'K', 'T']:  # T = 10
            ranks.append(10)
        else:
            ranks.append(int(rank))
    
    # Calculate score and count aces
    score = 0
    aces = 0
    for rank in ranks:
        if rank == 1:
            aces += 1
            score += 11
        else:
            score += rank
    
    # Adjust for aces to avoid busting
    while score > 21 and aces > 0:
        score -= 10
        aces -= 1
    
    # A usable ace is one that can be counted as 11 without busting
    # This means we have at least one ace and the score is <= 21
    has_usable_ace = aces > 0 and score <= 21
    
    return score, has_usable_ace

class MonteCarloBlackjackAgent:
    """
    Enhanced Monte Carlo agent for blackjack with plotting and hyperparameter sweeps
    """
    
    def __init__(self, num_episodes=500000, gamma=0.95, epsilon=0.1, alpha=0.05, first_visit=True):
        self.num_episodes = num_episodes
        self.gamma = gamma  
        self.epsilon = epsilon 
        self.alpha = alpha  
        self.first_visit = first_visit  
        
        
        self.Q = {}  # Q-value function Q(s,a)
        self.V = {}  # State value function V(s) = max_a Q(s,a)
        self.N = {}  # Visit counts N(s,a)
        self.policy = {}  # Optimal policy
        
        # Training tracking
        self.training_stats = {
            'episodes_played': 0,
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'episode_rewards': [],
            'rolling_winrate': [],
            'checkpoint_episodes': []
        }
        
        
        self.hyperparams = {
            'num_episodes': num_episodes,
            'gamma': gamma,
            'epsilon': epsilon,
            'alpha': alpha,
            'first_visit': first_visit
        }
    
    def get_state_key(self, state):
        """
        Create a unique key for the current game state
        state = (player_sum, dealer_showing, usable_ace)
        """
        player_sum, dealer_showing, usable_ace = state
        return (player_sum, dealer_showing, usable_ace)
    
    def epsilon_greedy_policy(self, state):
        """
        Epsilon-greedy policy for action selection using both Q-values and state values
        """
        state_key = self.get_state_key(state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.Q:
            self.Q[state_key] = {0: 0.0, 1: 0.0}
            self.N[state_key] = {0: 0, 1: 0}
            self.V[state_key] = 0.0
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            # Greedy action (choose action with highest Q-value)
            return max(self.Q[state_key], key=self.Q[state_key].get)
    
    def train(self, verbose=True, checkpoint_interval=10000):
        """
        Train the Monte Carlo agent with enhanced tracking and plotting
        """
        if verbose:
            print("🎯 Starting Enhanced Monte Carlo RL Training...")
            print(f"Training for {self.num_episodes} episodes")
            print(f"Parameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}, first_visit={self.first_visit}")
        
        # Initialize tracking
        episode_rewards = []
        rolling_winrates = []
        checkpoint_episodes = []
        
        # Progress bar
        pbar = tqdm(range(self.num_episodes), desc="Training", unit="episodes")
        
        for episode in pbar:
            # Create fresh environment for each episode
            env = rlcard.make('blackjack')
            state, _ = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards_list = []
            
            # Play one episode
            while not env.is_over():
                if env.get_player_id() == 0:  # Player's turn
                    
                    raw_obs = state['raw_obs']
                    player_sum, dealer_showing, usable_ace = decode_rlcard_state(raw_obs)
                    current_state = (player_sum, dealer_showing, usable_ace)
                    
                    # Choose action using epsilon-greedy policy
                    action = self.epsilon_greedy_policy(current_state)
                    
                  
                    episode_states.append(current_state)
                    episode_actions.append(action)
                    episode_rewards_list.append(0) 
                    
                    
                    state, _ = env.step(action)
                else:  
                    state, _ = env.step(0)  
            
           
            final_reward = env.get_payoffs()[0]
            if episode_rewards_list:
                episode_rewards_list[-1] = final_reward
            
            # Update Q-values using Monte Carlo method
            self.update_q_values(episode_states, episode_actions, episode_rewards_list)
            
            
            episode_rewards.append(final_reward)
            
            
            self.training_stats['episodes_played'] += 1
            if final_reward > 0:
                self.training_stats['wins'] += 1
            elif final_reward < 0:
                self.training_stats['losses'] += 1
            else:
                self.training_stats['ties'] += 1
            
            
            if (episode + 1) % checkpoint_interval == 0:
                # Compute state values
                self.compute_state_values()
                
                
                recent_episodes = episode_rewards[-checkpoint_interval:]
                wins = sum(1 for r in recent_episodes if r > 0)
                rolling_winrate = (wins / len(recent_episodes)) * 100
                rolling_winrates.append(rolling_winrate)
                checkpoint_episodes.append(episode + 1)
                
                
                pbar.set_postfix({
                    'Win Rate (Train)': f'{rolling_winrate:.1f}%',
                    'States': len(self.Q),
                    'Avg Reward': f'{np.mean(recent_episodes):.3f}'
                })
        
        pbar.close()
        
       
        self.training_stats['episode_rewards'] = episode_rewards
        self.training_stats['rolling_winrate'] = rolling_winrates
        self.training_stats['checkpoint_episodes'] = checkpoint_episodes
        
       
        self.compute_state_values()
        
       
        self.extract_policy()
        
        if verbose:
            print(" Enhanced Monte Carlo RL training completed!")
            print(f"Trained Q-values for {len(self.Q)} states")
            print(f"Episodes: {self.training_stats['episodes_played']}")
            print(f"Wins: {self.training_stats['wins']}, Losses: {self.training_stats['losses']}, Ties: {self.training_stats['ties']}")
            if rolling_winrates:
                print(f"Final rolling win rate (Train): {rolling_winrates[-1]:.1f}%")
    
    def update_q_values(self, states, actions, rewards):
        """
        Update Q-values using Monte Carlo method 
        """
       
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = self.gamma * G + reward
            returns.insert(0, G)
        
        if self.first_visit:
            # First-visit MC: update each (state, action) only on first occurrence
            seen_pairs = set()
            for i, (state, action) in enumerate(zip(states, actions)):
                state_key = self.get_state_key(state)
                pair = (state_key, action)
                
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    
                    
                    if state_key not in self.Q:
                        self.Q[state_key] = {0: 0.0, 1: 0.0}
                        self.N[state_key] = {0: 0, 1: 0}
                    
               
                    self.N[state_key][action] += 1
                    
                  
                    current_q = self.Q[state_key][action]
                    return_value = returns[i]
                    
                    # Q(s,a) = Q(s,a) + α * [G - Q(s,a)]
                    self.Q[state_key][action] = current_q + self.alpha * (return_value - current_q)
        else:
            
            for i, (state, action) in enumerate(zip(states, actions)):
                state_key = self.get_state_key(state)
                
               
                if state_key not in self.Q:
                    self.Q[state_key] = {0: 0.0, 1: 0.0}
                    self.N[state_key] = {0: 0, 1: 0}
                
                # Update visit count
                self.N[state_key][action] += 1
                
                # Update Q-value using learning rate α
                current_q = self.Q[state_key][action]
                return_value = returns[i]
                
                # Q(s,a) = Q(s,a) + α * [G - Q(s,a)]
                self.Q[state_key][action] = current_q + self.alpha * (return_value - current_q)
    
    def compute_state_values(self):
        """
        Compute state value function V(s) = max_a Q(s,a)
        """
        for state_key, q_values in self.Q.items():
            # V(s) = max_a Q(s,a)
            self.V[state_key] = max(q_values.values())
    
    def extract_policy(self):
        """
        Extract optimal policy from Q-values
        """
        for state_key, q_values in self.Q.items():
            # Choose action with highest Q-value
            best_action = max(q_values, key=q_values.get)
            self.policy[state_key] = best_action
    
    def get_action(self, player_score, dealer_card, has_ace=False):
        """
        Get the optimal action for a given state
        """
        # Convert to proper state format
        dealer_showing = dealer_card if isinstance(dealer_card, int) else 10 if dealer_card in ['J', 'Q', 'K'] else 1 if dealer_card == 'A' else int(dealer_card)
        usable_ace = 1 if has_ace else 0
        state = (player_score, dealer_showing, usable_ace)
        state_key = self.get_state_key(state)
        
        if state_key in self.policy:
            return self.policy[state_key]
        else:
            # Default strategy if state not found
            if player_score < 17:
                return 0  # Hit
            else:
                return 1  # Stand
    
    def save_strategy(self, filename="monte_carlo_strategy.pkl"):
        """
        Save the trained strategy to a file
        """
        strategy_data = {
            'Q': self.Q,
            'V': self.V,
            'policy': self.policy,
            'training_stats': self.training_stats,
            'num_episodes': self.num_episodes,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'alpha': self.alpha
        }
        
        with open(f'models/{filename}', 'wb') as f:
            pickle.dump(strategy_data, f)
        
        print(f" Monte Carlo RL strategy saved to models/{filename}")
    
    def load_strategy(self, filename="monte_carlo_strategy.pkl"):
        """
        Load a trained strategy from a file
        """
        if os.path.exists(f'models/{filename}'):
            with open(f'models/{filename}', 'rb') as f:
                strategy_data = pickle.load(f)
                
                self.Q = strategy_data.get('Q', {})
                self.V = strategy_data.get('V', {})
                self.policy = strategy_data.get('policy', {})
                self.training_stats = strategy_data.get('training_stats', {})
                self.num_episodes = strategy_data.get('num_episodes', 200000)
                self.gamma = strategy_data.get('gamma', 0.9)
                self.epsilon = strategy_data.get('epsilon', 0.1)
                self.alpha = strategy_data.get('alpha', 0.1)
            
            print(f"Monte Carlo RL strategy loaded from {filename}")
            print(f" Strategy contains {len(self.policy)} policy states")
            return True
        else:
            print(f"Strategy file {filename} not found")
            return False
    
    def create_state_value_heatmaps(self):
        """
        Create state value heatmaps matching Sutton & Barto Figure 5.2 style
        """
        if not self.V:
            print(" No state values available. Train the agent first.")
            return
        
       
        player_sums = list(range(12, 22))  # 12-21
        dealer_showings = list(range(1, 11))  # 1-10 (A=1, T/J/Q/K=10)
        
        hard_grid = np.zeros((len(player_sums), len(dealer_showings)))
        soft_grid = np.zeros((len(player_sums), len(dealer_showings)))
        
      
        for i, player_sum in enumerate(player_sums):
            for j, dealer_showing in enumerate(dealer_showings):
               
                hard_state = (player_sum, dealer_showing, False)
                if hard_state in self.V:
                    hard_grid[i, j] = self.V[hard_state]
                else:
                    hard_grid[i, j] = 0.0  
                
                # Soft totals (usable ace)
                soft_state = (player_sum, dealer_showing, True)
                if soft_state in self.V:
                    soft_grid[i, j] = self.V[soft_state]
                else:
                    soft_grid[i, j] = 0.0  
        
       
        all_values = np.concatenate([hard_grid.flatten(), soft_grid.flatten()])
        if len(all_values) > 0:
            vmin, vmax = np.min(all_values), np.max(all_values)
            # Normalize to [-1, +1]
            if vmax > vmin:
                hard_grid_norm = 2 * (hard_grid - vmin) / (vmax - vmin) - 1
                soft_grid_norm = 2 * (soft_grid - vmin) / (vmax - vmin) - 1
            else:
                hard_grid_norm = hard_grid
                soft_grid_norm = soft_grid
        else:
            hard_grid_norm = hard_grid
            soft_grid_norm = soft_grid
        
       
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
       
        im1 = ax1.imshow(hard_grid_norm, cmap='RdYlBu_r', aspect='auto', origin='lower', vmin=-1, vmax=1)
        ax1.set_title('No Usable Ace', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Dealer showing', fontsize=12)
        ax1.set_ylabel('Player sum', fontsize=12)
        ax1.set_xticks(range(len(dealer_showings)))
        ax1.set_xticklabels(dealer_showings)
        ax1.set_yticks(range(len(player_sums)))
        ax1.set_yticklabels(player_sums)
        
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('V(s)', fontsize=10)
        
     
        im2 = ax2.imshow(soft_grid_norm, cmap='RdYlBu_r', aspect='auto', origin='lower', vmin=-1, vmax=1)
        ax2.set_title('Usable Ace', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dealer showing', fontsize=12)
        ax2.set_ylabel('Player sum', fontsize=12)
        ax2.set_xticks(range(len(dealer_showings)))
        ax2.set_xticklabels(dealer_showings)
        ax2.set_yticks(range(len(player_sums)))
        ax2.set_yticklabels(player_sums)
        
      
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('V(s)', fontsize=10)
        
        plt.suptitle('Blackjack State-Value Function V(s) = max_a Q(s,a)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save 2D heatmap
        plt.savefig('visualizations/V_soft_vs_hard_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        
        from mpl_toolkits.mplot3d import Axes3D
        
       
        X, Y = np.meshgrid(dealer_showings, player_sums)
        
        
        fig = plt.figure(figsize=(20, 8))
        
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, hard_grid_norm, cmap='RdYlBu_r', alpha=0.8, vmin=-1, vmax=1)
        ax1.set_title('No Usable Ace (3D)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Dealer showing', fontsize=12)
        ax1.set_ylabel('Player sum', fontsize=12)
        ax1.set_zlabel('V(s)', fontsize=12)
        ax1.view_init(elev=30, azim=45)
        
       
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, soft_grid_norm, cmap='RdYlBu_r', alpha=0.8, vmin=-1, vmax=1)
        ax2.set_title('Usable Ace (3D)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dealer showing', fontsize=12)
        ax2.set_ylabel('Player sum', fontsize=12)
        ax2.set_zlabel('V(s)', fontsize=12)
        ax2.view_init(elev=30, azim=45)
        
        plt.suptitle('Blackjack State-Value Function V(s) - 3D Surface Plots', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
      
        plt.savefig('visualizations/V_soft_vs_hard_3d.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" State value visualizations saved:")
        print("   - visualizations/V_soft_vs_hard_heatmap.png (2D heatmaps)")
        print("   - visualizations/V_soft_vs_hard_3d.png (3D surfaces side-by-side)")
    
    
    def create_training_plots(self):
        """
        Create training performance plots
        """
        if not self.training_stats['checkpoint_episodes']:
            print("  No checkpoint data available for plotting")
            return
        
       
        checkpoint_rewards = []
        for episode in self.training_stats['checkpoint_episodes']:
            start_idx = max(0, episode - 10000)
            end_idx = episode
            recent_rewards = self.training_stats['episode_rewards'][start_idx:end_idx]
            avg_reward = np.mean(recent_rewards)
            checkpoint_rewards.append(avg_reward)
        
       
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Average reward plot
        ax1.plot(self.training_stats['checkpoint_episodes'], checkpoint_rewards, 'b-', linewidth=2)
        ax1.set_title('Average Reward per 10k Episodes')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.grid(True, alpha=0.3)
        
        # Win rate plot
        ax2.plot(self.training_stats['checkpoint_episodes'], self.training_stats['rolling_winrate'], 'r-', linewidth=2)
        ax2.set_title('Rolling Win Rate (Train) per 10k Episodes')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Win Rate (Train) (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/training_reward_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Training plot saved as visualizations/training_reward_curve.png")
    
    def save_training_history(self):
        """
        Save training history to CSV
        """
        if not self.training_stats['checkpoint_episodes']:
            print("  No checkpoint data available for CSV export")
            return
       
        checkpoint_rewards = []
        for episode in self.training_stats['checkpoint_episodes']:
            start_idx = max(0, episode - 10000)
            end_idx = episode
            recent_rewards = self.training_stats['episode_rewards'][start_idx:end_idx]
            avg_reward = np.mean(recent_rewards)
            checkpoint_rewards.append(avg_reward)
        
       
        df = pd.DataFrame({
            'episode': self.training_stats['checkpoint_episodes'],
            'avg_reward': checkpoint_rewards,
            'rolling_winrate': self.training_stats['rolling_winrate']
        })
        
        df.to_csv('results/training_history.csv', index=False)
        print(" Training history saved as results/training_history.csv")


class TrainingEnvironment:
    """
    Environment for training and testing AI agents
    """
    
    def __init__(self):
        self.env = rlcard.make('blackjack')
    
    def test_agent(self, agent, num_games=1000, verbose=True):
        """
        Test an agent's performance using RLCard
        """
        wins = 0
        losses = 0
        ties = 0
        
        if verbose:
            print(f"Testing agent with {num_games} games...")
        
        for _ in range(num_games):
            
            result = self.simulate_single_game(agent)
            
            if result == 1:
                wins += 1
            elif result == 0:
                losses += 1
            else:
                ties += 1
        
        win_rate = wins / num_games * 100
        
        if verbose:
            print(f" Test Results ({num_games} games):")
            print(f"   Wins: {wins} ({wins/num_games*100:.1f}%)")
            print(f"   Losses: {losses} ({losses/num_games*100:.1f}%)")
            print(f"   Ties: {ties} ({ties/num_games*100:.1f}%)")
            print(f"   Win Rate (Test): {win_rate:.1f}%")
        
        return {
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_rate': win_rate
        }
    
    def simulate_single_game(self, agent):
        """
        Simulate a single blackjack game using RLCard
        """
       
        state, _ = self.env.reset()
        
        while not self.env.is_over():
            
            raw_obs = state['raw_obs']
            
            # Decode RLCard state to (player_sum, dealer_showing, usable_ace)
            player_sum, dealer_showing, usable_ace = decode_rlcard_state(raw_obs)
            
            
            action = agent.get_action(player_sum, dealer_showing, usable_ace)
            
            
            state, _ = self.env.step(action)
        
        
        payoffs = self.env.get_payoffs()
        final_reward = payoffs[0]  
        
        # Convert reward to win/loss/tie
        if final_reward > 0:
            return 1 
        elif final_reward < 0:
            return 0  
        else:
            return 0.5  

def hyperparameter_sweep():
    """
    Perform hyperparameter sweep over alpha and gamma values
    """
    print(" Starting Hyperparameter Sweep...")
    print("=" * 60)
    
    
    alphas = [0.01, 0.05, 0.1]
    gammas = [0.9, 0.95, 0.99]
    num_episodes = 100000
    eval_games = 50000
    
    results = []
    
    for alpha in alphas:
        for gamma in gammas:
            print(f"\n Testing α={alpha}, γ={gamma}")
            print("-" * 40)
            
            # Create and train agent
            agent = MonteCarloBlackjackAgent(
                num_episodes=num_episodes,
                gamma=gamma,
                epsilon=0.1,
                alpha=alpha,
                first_visit=True
            )
            
            
            agent.train(verbose=False)
            
           
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0  
            
            env = TrainingEnvironment()
            eval_results = env.test_agent(agent, num_games=eval_games)
            
            
            agent.epsilon = original_epsilon
            
            
            result = {
                'alpha': alpha,
                'gamma': gamma,
                'win_rate': eval_results['win_rate'],
                'wins': eval_results['wins'],
                'losses': eval_results['losses'],
                'ties': eval_results['ties'],
                'states_learned': len(agent.Q)
            }
            results.append(result)
            
            print(f"   Win Rate (Test): {eval_results['win_rate']:.1f}%")
            print(f"   States Learned: {len(agent.Q)}")
    
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('win_rate', ascending=False)
    
   
    df_results.to_csv('results/grid_search_results.csv', index=False)
    
  
    print("\n HYPERPARAMETER SWEEP RESULTS")
    print("=" * 60)
    print(df_results.to_string(index=False, float_format='%.2f'))
    print("\n Results saved to results/grid_search_results.csv")
    
    return df_results


def main():
    """
    Main function to train and test the enhanced Monte Carlo agent
    """
    print(" Enhanced Monte Carlo RL Blackjack Training")
    print("=" * 60)
    
   
    random.seed(42)
    np.random.seed(42)
    
  
    mc_agent = MonteCarloBlackjackAgent(
        num_episodes=1000000,
        gamma=1.0,
        epsilon=0.1,
        alpha=0.05,
        first_visit=True
    )
    
    
    print(" Starting main training run...")
    mc_agent.train(verbose=True, checkpoint_interval=10000)
    
    
    print("\nCreating visualizations...")
    mc_agent.create_state_value_heatmaps()
    mc_agent.create_training_plots()
    mc_agent.save_training_history()
    
    
    print("\n Testing trained agent...")
    env = TrainingEnvironment()
    mc_results = env.test_agent(mc_agent, num_games=50000)
    
   
    print("\n FINAL RESULTS")
    print("=" * 50)
    print("Monte Carlo Agent:")
    print(f"   Win Rate (Test): {mc_results['win_rate']:.1f}%")
    print(f"   Wins: {mc_results['wins']}, Losses: {mc_results['losses']}, Ties: {mc_results['ties']}")
    print("   Training: Enhanced Monte Carlo RL")
    print(f"   States Learned: {len(mc_agent.Q)}")
    print(f"   Episodes: {mc_agent.num_episodes}")
    print(f"   Parameters: α={mc_agent.alpha}, γ={mc_agent.gamma}, ε={mc_agent.epsilon}")
    print(f"   First-visit MC: {mc_agent.first_visit}")
    
   
    mc_agent.save_strategy()
    
    print("\n Enhanced training completed!")
    print(" Generated files:")
    print("    models/")
    print("      - monte_carlo_strategy.pkl (trained model)")
    print("    visualizations/")
    print("      - V_soft_vs_hard_heatmap.png (2D heatmaps)")
    print("      - V_soft_vs_hard_3d.png (3D surfaces)")
    print("      - training_reward_curve.png (reward & win rate progression)")
    print("    results/")
    print("      - training_history.csv (training data)")
    print("      - grid_search_results.csv (hyperparameter results)")
    
   
    print("\nRun hyperparameter sweep? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            hyperparameter_sweep()
    except (EOFError, KeyboardInterrupt):
        print("Skipping hyperparameter sweep.")


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Enhanced Monte Carlo RL Blackjack Training')
    parser.add_argument('--episodes', type=int, default=1000000, help='Number of training episodes')
    parser.add_argument('--alpha', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for epsilon-greedy')
    parser.add_argument('--first-visit', action='store_true', default=True, help='Use first-visit MC')
    parser.add_argument('--eval-games', type=int, default=50000, help='Number of evaluation games')
    parser.add_argument('--grid-search', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
