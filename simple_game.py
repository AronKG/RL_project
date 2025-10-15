import rlcard
import numpy as np

def play_blackjack():
   
    print("🃏 Welcome to Blackjack!")
    print("=" * 40)
    
    
    env = rlcard.make('blackjack')
    
    while True:
        print("\n🎮 New Game")
        print("-" * 20)
        
        # Reset environment
        state, player_id = env.reset()
        
      
        print(f"Your cards: {state['raw_obs']['player0 hand']}")
        print(f"Dealer's card: {state['raw_obs']['dealer hand']}")
        print(f"Your score: {state['obs'][0]}")
        print(f"Dealer's score: {state['obs'][1]}")
        
        # Player's turn
        while not env.is_over():
            legal_actions = list(state['legal_actions'].keys())
            
            action_names = {0: "Hit", 1: "Stand"}
            print(f"\nLegal actions: {[action_names[action] for action in legal_actions]}")
            
            # Get player input
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
                except KeyboardInterrupt:
                    print("\nGame ended by user.")
                    return
            
            # Make move
            state, player_id = env.step(action)
            action_name = "Hit" if action == 0 else "Stand"
            
            print(f"You chose: {action_name}")
            print(f"Your score: {state['obs'][0]}")
            print(f"Dealer's score: {state['obs'][1]}")
            
            if env.is_over():
                break
        
        # Game over
        payoffs = env.get_payoffs()
        player_reward = payoffs[0]
        
        if player_reward > 0:
            result = "🎉 You Win!"
        elif player_reward < 0:
            result = "😞 You Lose!"
        else:
            result = "🤝 It's a Tie!"
        
        print(f"\n{result}")
        print(f"Final reward: {player_reward}")
        
        # Ask to play again
        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again != 'y':
            break
    
    print("\nThanks for playing! 👋")

if __name__ == "__main__":
    play_blackjack()
