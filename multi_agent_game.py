#!/usr/bin/env python3
"""
Multi-Agent Blackjack Simulation
Two trained Monte Carlo agents vs. one dealer (RLCard environment)
Author: Parna Saeidpour
"""

import rlcard
from ai_training import MonteCarloBlackjackAgent, decode_rlcard_state 
import numpy as np

def simulate_multi_agent_game(agent1, agent2, num_games=1000):
    """
    Simulate Blackjack with two trained agents vs. one dealer.
    Each agent plays independently against the dealer.
    """
    env = rlcard.make('blackjack')
    stats = {'agent1_wins': 0, 'agent2_wins': 0, 'dealer_wins': 0, 'ties': 0}

    for game in range(num_games):
        state, _ = env.reset()
        done = False
        current_player = env.get_player_id()

        # Track rewards per player
        while not env.is_over():
            current_player = env.get_player_id()
            raw_obs = state['raw_obs']

            # Decode current player's state
            player_sum, dealer_showing, usable_ace = decode_rlcard_state(raw_obs)

            if current_player == 0:
                # Agent 1 plays
                current_state = (player_sum, dealer_showing, usable_ace)
                action = agent1.epsilon_greedy_policy(current_state)
            elif current_player == 1:
                # Agent 2 plays
                current_state = (player_sum, dealer_showing, usable_ace)
                action = agent2.epsilon_greedy_policy(current_state)
            else:
                # Dealer's turn
                action = 0  # Dealer follows fixed rule

            state, _ = env.step(action)

        # After episode ends
        payoffs = env.get_payoffs()  # [player_reward] - single player vs dealer
        player_reward = payoffs[0]
        
        # Both agents get the same reward since they're playing the same game
        agent1_r = player_reward
        agent2_r = player_reward

        # Determine outcomes for each agent
        if agent1_r > 0:
            stats['agent1_wins'] += 1
        elif agent1_r < 0:
            stats['dealer_wins'] += 1
        else:
            stats['ties'] += 1

        if agent2_r > 0:
            stats['agent2_wins'] += 1
        elif agent2_r < 0:
            stats['dealer_wins'] += 1
        else:
            stats['ties'] += 1

    # Compute final win rates
    total_games = num_games * 2
    print("\nMULTI-AGENT SIMULATION RESULTS ")
    print("=" * 50)
    print(f"Agent 1 Win Rate: {stats['agent1_wins'] / num_games * 100:.2f}%")
    print(f"Agent 2 Win Rate: {stats['agent2_wins'] / num_games * 100:.2f}%")
    print(f"Dealer Win Rate: {stats['dealer_wins'] / total_games * 100:.2f}%")
    print(f"Ties: {stats['ties']} total")
    print(f"Total Games: {total_games}")
    print("=" * 50)

    return stats


def main():
    print(" Loading trained Monte Carlo agent for multi-agent simulation...")

    # Load trained Monte Carlo agents
    agent1 = MonteCarloBlackjackAgent()
    if not agent1.load_strategy('monte_carlo_strategy.pkl'):
        print(" Failed to load agent1. Please train the model first.")
        return
# we can add q-learning agent here
    agent2 = MonteCarloBlackjackAgent()
    if not agent2.load_strategy('monte_carlo_strategy.pkl'):
        print(" Failed to load agent2. Please train the model first.")
        return

    # Run multi-agent simulation
    simulate_multi_agent_game(agent1, agent2, num_games=5000)


if __name__ == "__main__":
    main()