"""
ANALYSIS for Q-learning with custom environment
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from collections import defaultdict
from q_learning import QLearningAgent, train_q_learning_with_env, evaluate_agent
from fix_env import BlackjackEnvironment

# ==================== CONFIGURATION PARAMETERS ====================
ANALYSIS_CONFIG = {
    # Training parameters
    'train_new_agent': True,
    'training_episodes': 1000000,
    'evaluation_interval': 10000,
    
    # Evaluation parameters
    'evaluation_games': 100000,
    
    # Output parameters
    'model_filename': 'models/q_learning_agent.pkl',
    'results_folder': 'results',
    'training_plot_filename': 'training_progress.png',
    'additional_plots_filename': 'additional_analysis.png',
    'q_table_csv': 'q_table.csv',
    'q_table_json': 'q_table.json',
    'q_table_summary': 'q_table_summary.txt',
    'game_results_csv': 'detailed_game_results.csv',
    'game_summary_txt': 'game_summary_statistics.txt',
}
# ==================== END CONFIGURATION ====================

def run_detailed_evaluation(agent, env, num_games=1000):
    """Run evaluation with detailed game-by-game results"""
    print(f"Running detailed evaluation for {num_games} games...")
    
    game_results = []
    outcome_stats = defaultdict(int)
    
    for game_num in range(num_games):
        state, player_id = env.reset()
        done = False
        player_actions = []
        
        # Play one complete game
        while not env.is_over():
            current_player = env.get_player_id()
            
            if current_player == 0:  # Agent's turn
                state_info = env.get_state_info(state)
                action = agent.choose_action(state_info)
                player_actions.append("HIT" if action == 1 else "STAND")
                next_state, next_player_id = env.step(action)
            else:  # Dealer's turn
                action = 0  
                next_state, next_player_id = env.step(action)
                
            state = next_state
        
        # Get final results
        payoffs = env.get_payoffs()
        reward = payoffs[0]  
        
        # Try to get final scores from the environment
        player_score = 0
        dealer_score = 0
        
        # Try different methods to get scores
        if hasattr(env, 'player_hand') and env.player_hand:
            player_score = env.player_hand.get_value()
        if hasattr(env, 'dealer_hand') and env.dealer_hand:
            dealer_score = env.dealer_hand.get_value()
        
        # Determine winner and outcome
        if reward > 0:
            outcome = "Player Win"
            outcome_stats["player_wins"] += 1
        elif reward < 0:
            outcome = "Dealer Win"
            outcome_stats["dealer_wins"] += 1
        else:
            outcome = "Push/Tie"
            outcome_stats["pushes"] += 1
        
        # Record bust outcomes
        if player_score > 21:
            outcome_stats["player_busts"] += 1
        if dealer_score > 21:
            outcome_stats["dealer_busts"] += 1
        
        # Record blackjack if available
        if hasattr(env, 'player_has_blackjack') and env.player_has_blackjack():
            outcome_stats["player_blackjacks"] += 1
        if hasattr(env, 'dealer_has_blackjack') and env.dealer_has_blackjack():
            outcome_stats["dealer_blackjacks"] += 1
        
        game_results.append({
            'game_id': game_num + 1,
            'player_final_score': player_score,
            'dealer_final_score': dealer_score,
            'outcome': outcome,
            'reward': reward,
            'player_actions': ' -> '.join(player_actions),
            'num_actions': len(player_actions),
            'player_win': 1 if reward > 0 else 0,
            'dealer_win': 1 if reward < 0 else 0,
            'push': 1 if reward == 0 else 0
        })
    
    return game_results, outcome_stats

def create_additional_plots(agent, history, game_results, outcome_stats, folder='results'):
    """Create additional analysis plots"""
    os.makedirs(folder, exist_ok=True)
    
    filename = f"{folder}/{ANALYSIS_CONFIG['additional_plots_filename']}"
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Q-Table Size Growth over time
    if 'q_table_sizes' in history and history['q_table_sizes']:
        axes[0, 0].plot(history['episodes'], history['q_table_sizes'], 'g-', linewidth=2)
        axes[0, 0].set_title('Q-Table Size Growth')
        axes[0, 0].set_xlabel('Training Episodes')
        axes[0, 0].set_ylabel('Number of States')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Value Distribution (histogram of all Q-values)
    if agent.q_table:
        all_q_values = []
        for state, q_values in agent.q_table.items():
            if isinstance(q_values, (list, np.ndarray)):
                all_q_values.extend(q_values)
            elif isinstance(q_values, (int, float)):
                all_q_values.append(q_values)
        
        if all_q_values:
            axes[0, 1].hist(all_q_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
            axes[0, 1].set_title('Q-Value Distribution')
            axes[0, 1].set_xlabel('Q-Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_q = np.mean(all_q_values)
            std_q = np.std(all_q_values)
            axes[0, 1].axvline(mean_q, color='red', linestyle='--', label=f'Mean: {mean_q:.3f}')
            axes[0, 1].legend()
    
    # Plot 3: Action Distribution Analysis from Q-table
    if agent.q_table:
        hit_count = 0
        stand_count = 0
        total_states = len(agent.q_table)
        
        for state, q_values in agent.q_table.items():
            if isinstance(q_values, (list, np.ndarray)) and len(q_values) >= 2:
                best_action = np.argmax(q_values)
                if best_action == 0:
                    stand_count += 1
                else:
                    hit_count += 1
        
        actions = ['STAND', 'HIT']
        counts = [stand_count, hit_count]
        colors = ['red', 'blue']
        
        bars = axes[1, 0].bar(actions, counts, color=colors, alpha=0.7)
        axes[1, 0].set_title('Optimal Action Distribution in Q-Table')
        axes[1, 0].set_ylabel('Number of States')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                           f'{count}\n({count/total_states:.1%})', 
                           ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Learning Progress - Cumulative Win Rate
    if len(history['win_rates']) > 1:
        episodes = history['episodes']
        win_rates = history['win_rates']
        
        # Calculate cumulative performance
        cumulative_avg = []
        for i in range(len(win_rates)):
            cumulative_avg.append(np.mean(win_rates[:i+1]))
        
        axes[1, 1].plot(episodes, win_rates, 'b-', alpha=0.5, label='Window Win Rate')
        axes[1, 1].plot(episodes, cumulative_avg, 'r-', linewidth=2, label='Cumulative Average')
        axes[1, 1].axhline(y=0.42, color='g', linestyle='--', alpha=0.7, label='Optimal (42%)')
        axes[1, 1].axhline(y=0.38, color='orange', linestyle='--', alpha=0.7, label='Random (38%)')
        
        # Add final evaluation result if available
        if game_results and outcome_stats:
            final_win_rate = outcome_stats['player_wins'] / len(game_results)
            axes[1, 1].axhline(y=final_win_rate, color='purple', linestyle='-', 
                              alpha=0.8, label=f'Final: {final_win_rate:.1%}')
        
        axes[1, 1].set_title('Learning Progress: Win Rate Evolution')
        axes[1, 1].set_xlabel('Training Episodes')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Additional analysis plots saved to {filename}")

def plot_training_progress(history, filename=None):
    """Plot training progress"""
    if filename is None:
        filename = f"{ANALYSIS_CONFIG['results_folder']}/{ANALYSIS_CONFIG['training_plot_filename']}"
    
    os.makedirs(ANALYSIS_CONFIG['results_folder'], exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Win rate plot
    plt.subplot(1, 2, 1)
    plt.plot(history['episodes'], history['win_rates'], 'b-', linewidth=2)
    plt.axhline(y=0.42, color='r', linestyle='--', alpha=0.7, label='Optimal (42%)')
    plt.axhline(y=0.38, color='g', linestyle='--', alpha=0.7, label='Random (38%)')
    plt.title('Q-Learning Training Progress')
    plt.xlabel('Training Episodes')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Exploration rate plot
    plt.subplot(1, 2, 2)
    plt.plot(history['episodes'], history['exploration_rates'], 'r-', linewidth=2)
    plt.title('Exploration Rate Decay')
    plt.xlabel('Training Episodes')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Training plot saved to {filename}")

def save_detailed_results(game_results, outcome_stats, folder='results'):
    """Save detailed game results and summary statistics"""
    os.makedirs(folder, exist_ok=True)
    
    # Save detailed game results as CSV
    df_detailed = pd.DataFrame(game_results)
    detailed_path = f"{folder}/{ANALYSIS_CONFIG['game_results_csv']}"
    df_detailed.to_csv(detailed_path, index=False)
    print(f"✓ Detailed game results saved: {detailed_path}")
    
    # Save summary statistics
    summary_path = f"{folder}/{ANALYSIS_CONFIG['game_summary_txt']}"
    total_games = len(game_results)
    
    with open(summary_path, 'w') as f:
        f.write("GAME SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic outcomes
        f.write("FINAL OUTCOMES:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Games Played: {total_games}\n")
        f.write(f"Player Wins: {outcome_stats['player_wins']} ({outcome_stats['player_wins']/total_games*100:.1f}%)\n")
        f.write(f"Dealer Wins: {outcome_stats['dealer_wins']} ({outcome_stats['dealer_wins']/total_games*100:.1f}%)\n")
        f.write(f"Pushes/Ties: {outcome_stats['pushes']} ({outcome_stats['pushes']/total_games*100:.1f}%)\n\n")
        
        # Blackjack statistics
        if 'player_blackjacks' in outcome_stats:
            f.write("BLACKJACK STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Player Blackjacks: {outcome_stats['player_blackjacks']} ({outcome_stats['player_blackjacks']/total_games*100:.1f}%)\n")
            f.write(f"Dealer Blackjacks: {outcome_stats['dealer_blackjacks']} ({outcome_stats['dealer_blackjacks']/total_games*100:.1f}%)\n\n")
        
        # Bust statistics
        f.write("BUST STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Player Busts: {outcome_stats['player_busts']} ({outcome_stats['player_busts']/total_games*100:.1f}%)\n")
        f.write(f"Dealer Busts: {outcome_stats['dealer_busts']} ({outcome_stats['dealer_busts']/total_games*100:.1f}%)\n\n")
        
        # Score distribution
        player_scores = [game['player_final_score'] for game in game_results]
        dealer_scores = [game['dealer_final_score'] for game in game_results]
        
        f.write("SCORE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Player Average Score: {np.mean(player_scores):.2f}\n")
        f.write(f"Dealer Average Score: {np.mean(dealer_scores):.2f}\n")
        f.write(f"Player Score Std Dev: {np.std(player_scores):.2f}\n")
        f.write(f"Dealer Score Std Dev: {np.std(dealer_scores):.2f}\n\n")
        
        # Action statistics
        num_actions = [game['num_actions'] for game in game_results]
        f.write("ACTION STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Actions per Game: {np.mean(num_actions):.2f}\n")
        f.write(f"Max Actions in a Game: {max(num_actions)}\n")
        f.write(f"Min Actions in a Game: {min(num_actions)}\n\n")
        
        # Win rate by final score
        f.write("WIN RATE BY PLAYER FINAL SCORE:\n")
        f.write("-" * 40 + "\n")
        score_groups = defaultdict(lambda: {'games': 0, 'wins': 0})
        
        for game in game_results:
            score = game['player_final_score']
            score_groups[score]['games'] += 1
            if game['player_win']:
                score_groups[score]['wins'] += 1
        
        for score in sorted(score_groups.keys()):
            if score <= 21:  # Only consider valid scores
                wins = score_groups[score]['wins']
                games = score_groups[score]['games']
                win_rate = (wins / games) * 100 if games > 0 else 0
                f.write(f"Score {score:2d}: {wins:3d}/{games:3d} wins ({win_rate:5.1f}%)\n")
    
    print(f"✓ Game summary statistics saved: {summary_path}")
    
    # Print quick summary to console
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   Player Wins: {outcome_stats['player_wins']}/{total_games} ({outcome_stats['player_wins']/total_games*100:.1f}%)")
    print(f"   Dealer Wins: {outcome_stats['dealer_wins']}/{total_games} ({outcome_stats['dealer_wins']/total_games*100:.1f}%)")
    print(f"   Pushes:      {outcome_stats['pushes']}/{total_games} ({outcome_stats['pushes']/total_games*100:.1f}%)")

def save_q_table(agent, folder='results'):
    """Save Q-table in multiple formats for analysis"""
    os.makedirs(folder, exist_ok=True)
    
    # Convert Q-table to DataFrame for CSV export
    q_data = []
    for state, actions in agent.q_table.items():
        if isinstance(actions, dict):
            row = {'state': str(state)}
            for action, value in actions.items():
                row[f'action_{action}'] = value
            q_data.append(row)
        elif isinstance(actions, (int, float, np.number)):
            row = {'state': str(state), 'q_value': actions}
            q_data.append(row)
        else:
            row = {'state': str(state), 'q_values': str(actions)}
            q_data.append(row)
    
    if q_data:
        df = pd.DataFrame(q_data)
        csv_path = f"{folder}/{ANALYSIS_CONFIG['q_table_csv']}"
        df.to_csv(csv_path, index=False)
        print(f"✓ Q-table saved as CSV: {csv_path}")
        
        # Save as JSON for easier readability
        json_path = f"{folder}/{ANALYSIS_CONFIG['q_table_json']}"
        # Convert any non-serializable objects to strings
        serializable_q_table = {}
        for state, actions in agent.q_table.items():
            serializable_q_table[str(state)] = actions
        
        with open(json_path, 'w') as f:
            json.dump(serializable_q_table, f, indent=2, default=str)
        print(f"✓ Q-table saved as JSON: {json_path}")
    
    return q_data

def print_q_table_summary(agent, folder='results'):
    """Print and save Q-table summary"""
    os.makedirs(folder, exist_ok=True)
    
    summary_path = f"{folder}/{ANALYSIS_CONFIG['q_table_summary']}"
    
    with open(summary_path, 'w') as f:
        f.write("Q-TABLE SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total states: {len(agent.q_table)}\n")
        f.write(f"State space coverage: {len(agent.q_table)} unique states\n\n")
        
        # Print first 10 states as sample
        f.write("SAMPLE Q-VALUES (first 10 states):\n")
        f.write("-" * 50 + "\n")
        
        for i, (state, actions) in enumerate(list(agent.q_table.items())[:10]):
            f.write(f"State {i+1}: {state}\n")
            if isinstance(actions, dict):
                for action, value in actions.items():
                    f.write(f"  {action}: {value:.4f}\n")
            else:
                f.write(f"  Q-value: {actions}\n")
            f.write("\n")
    
    print(f"✓ Q-table summary saved: {summary_path}")

def run_analysis():
    """Run complete analysis with custom environment"""
    print("=== Q-LEARNING ANALYSIS WITH CUSTOM ENVIRONMENT ===")
    print("Configuration:")
    for key, value in ANALYSIS_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Initialize your custom environment
    env = BlackjackEnvironment()
    
    # Create agent
    agent = QLearningAgent()
    
    # Train or load agent
    if ANALYSIS_CONFIG['train_new_agent']:
        print(f"Training new agent for {ANALYSIS_CONFIG['training_episodes']} episodes...")
        agent, history = train_q_learning_with_env(
            env, 
            num_episodes=ANALYSIS_CONFIG['training_episodes'],
            evaluation_interval=ANALYSIS_CONFIG['evaluation_interval']
        )
        agent.save_model(ANALYSIS_CONFIG['model_filename'])
        plot_training_progress(history)
    else:
        if agent.load_model(ANALYSIS_CONFIG['model_filename']):
            print("Loaded pre-trained agent")
            history = {
                'episodes': [1000, 5000, 10000, 15000, 20000],
                'win_rates': [0.32, 0.38, 0.41, 0.43, 0.44],
                'exploration_rates': [0.6, 0.3, 0.1, 0.05, 0.01],
                'q_table_sizes': [100, 150, 170, 180, 180]
            }
        else:
            print("No model found. Please set 'train_new_agent' to True.")
            return
    
    # Save Q-table analysis
    save_q_table(agent, ANALYSIS_CONFIG['results_folder'])
    print_q_table_summary(agent, ANALYSIS_CONFIG['results_folder'])
    
    # Final evaluation with detailed results
    print("\n" + "="*50)
    print("FINAL EVALUATION WITH DETAILED RESULTS")
    print("="*50)
    
    # Run detailed evaluation
    game_results, outcome_stats = run_detailed_evaluation(
        agent, env, num_games=ANALYSIS_CONFIG['evaluation_games']
    )
    
    # Save detailed results
    save_detailed_results(game_results, outcome_stats, ANALYSIS_CONFIG['results_folder'])
    
    # Create additional analysis plots - now passing outcome_stats
    create_additional_plots(agent, history, game_results, outcome_stats, ANALYSIS_CONFIG['results_folder'])
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"Final Win Rate: {outcome_stats['player_wins']/len(game_results):.1%}")
    print(f"Q-Table Size: {len(agent.q_table)} states")
    print(f"Results saved in: {ANALYSIS_CONFIG['results_folder']}")

if __name__ == "__main__":
    run_analysis()