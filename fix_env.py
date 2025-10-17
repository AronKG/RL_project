"""
BLACKJACK ENVIRONMENT
Pure game environment without any RL algorithms
"""

import rlcard
import numpy as np

class BlackjackEnvironment:
    def __init__(self):
        self.env = rlcard.make('blackjack')
        self.num_actions = self.env.num_actions
        
    def reset(self):
        """Reset the game and return initial state"""
        state, player_id = self.env.reset()
        return state, player_id
    
    def step(self, action):
        """Take an action and return next state, player_id"""
        return self.env.step(action)
    
    def is_over(self):
        """Check if game is over"""
        return self.env.is_over()
    
    def get_payoffs(self):
        """Get final rewards"""
        return self.env.get_payoffs()
    
    def get_player_id(self):
        """Get current player ID"""
        return self.env.get_player_id()
    
    def get_state_info(self, state):
        """Extract useful information from state"""
        obs = state['obs']
        raw_obs = state['raw_obs']
        
        return {
            'player_value': int(obs[0]),
            'dealer_showing': int(obs[1]),
            'player_hand': raw_obs['player0 hand'],
            'dealer_hand': raw_obs['dealer hand'],
            'legal_actions': list(state['legal_actions'].keys())
        }
    
    def play_manual_game(self):
        """Play a manual game for testing"""
        print("🃏 MANUAL BLACKJACK GAME")
        print("=" * 40)
        
        state, player_id = self.reset()
        state_info = self.get_state_info(state)
        
        print(f"Your cards: {state_info['player_hand']}")
        print(f"Dealer shows: {state_info['dealer_hand'][0]}")
        print(f"Your score: {state_info['player_value']}")
        
        while not self.is_over():
            current_player = self.get_player_id()
            
            if current_player == 0:  # Player's turn
                legal_actions = state_info['legal_actions']
                action_names = {0: "Hit", 1: "Stand"}
                
                print(f"\nLegal actions: {[action_names[a] for a in legal_actions]}")
                
                while True:
                    try:
                        choice = input("Choose action (0=Hit, 1=Stand): ").strip()
                        action = int(choice)
                        if action in legal_actions:
                            break
                        else:
                            print("Invalid action. Please choose 0 or 1.")
                    except ValueError:
                        print("Please enter a number (0 or 1).")
                
                # Take action
                state, player_id = self.step(action)
                state_info = self.get_state_info(state)
                
                print(f"You chose: {action_names[action]}")
                print(f"Your score: {state_info['player_value']}")
                
            else:  # Dealer's turn
                print("\nDealer's turn...")
                state, player_id = self.step(0)
                state_info = self.get_state_info(state)
        
        # Game over
        payoffs = self.get_payoffs()
        player_reward = payoffs[0]
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Your final hand: {state_info['player_hand']} = {state_info['player_value']}")
        print(f"Dealer's final hand: {state_info['dealer_hand']} = {self.get_state_info(state)['dealer_showing']}")
        
        if player_reward > 0:
            print("🎉 YOU WIN!")
        elif player_reward < 0:
            print("💀 YOU LOSE!")
        else:
            print("🤝 TIE!")
        
        return player_reward

def test_environment():
    """Test that the environment works correctly"""
    print("=== TESTING BLACKJACK ENVIRONMENT ===")
    env = BlackjackEnvironment()
    
    # Test basic functionality
    state, player_id = env.reset()
    state_info = env.get_state_info(state)
    
    print("✓ Environment initialized successfully")
    print(f"Initial state: Player={state_info['player_value']}, Dealer shows={state_info['dealer_showing']}")
    print(f"Legal actions: {state_info['legal_actions']}")
    
    # Test a few automated games
    print("\n=== TESTING AUTOMATED GAMEPLAY ===")
    for game in range(3):
        print(f"\nGame {game + 1}:")
        state, player_id = env.reset()
        
        while not env.is_over():
            current_player = env.get_player_id()
            state_info = env.get_state_info(state)
            
            if current_player == 0:
                # Simple strategy
                action = 1 if state_info['player_value'] > 17 else 0
                state, player_id = env.step(action)
            else:
                state, player_id = env.step(0)
        
        payoffs = env.get_payoffs()
        final_state_info = env.get_state_info(state)
        print(f"Final: Player={final_state_info['player_value']}, Dealer={final_state_info['dealer_showing']}, Reward={payoffs[0]}")

if __name__ == "__main__":
    test_environment()
    
    # Uncomment to play manual game
    # env = BlackjackEnvironment()
    # env.play_manual_game()