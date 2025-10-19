# Imports
import rlcard
from rlcard.utils.utils import print_card
from q_learning import QLearningAgent
from mc import MonteCarloBlackjackAgent, decode_rlcard_state
import sys
import shutil


def sep(title=None, width=None):
    """Print a consistent separator with optional centered title.
    """
    if width is None:
        try:
            width = shutil.get_terminal_size().columns
        except Exception:
            width = 60
    if title:
        title = f" {title} "
        fill = (width - len(title)) // 2
        print('-' * max(0, fill) + title + '-' * max(0, width - len(title) - fill))
    else:
        print('-' * width)


def show_hand(player_id, state, label=None):
    """Print a player's hand with a label. Uses print_card when possible.

    state: the RLCard state dict
    label: optional string label (e.g., 'Your hand', 'Q-agent hand')
    """
    if label is None:
        label = f'Player {player_id} hand'

    # Extract hand from state through raw_obs
    hand = None
    raw = state.get('raw_obs', {})
    hand = raw.get(f'player{player_id} hand') or raw.get('hand')

    print(f"{label}:")
    if hand is None:
        print('  (hand not available)')
        return
    # Normalize hand elements (convert card objects to short strings) and try pretty print
    #norm_hand = normalize_hand(hand)
    if isinstance(hand, list) and len(hand) > 0:
        print(hand)
        print_card(hand)
        return

    # Fallback to plain print
    print('  ', hand)


def agent_label(agent, idx=None):
    """Return a short label for the agent instance."""

    if isinstance(agent, InteractiveHumanAgent):
        return 'Human'
    if isinstance(agent, QAgentAdapter) or (hasattr(agent, 'qagent') and isinstance(getattr(agent, 'qagent'), QLearningAgent)):
        return 'Q-Learning'
    if isinstance(agent, MCAgentAdapter) or (hasattr(agent, 'mcagent') and isinstance(getattr(agent, 'mcagent'), MonteCarloBlackjackAgent)):
        return 'MonteCarlo'
    # Fallback: if agent has a class name
    try:
        return agent.__class__.__name__
    except Exception:
        return f'Player{idx}' if idx is not None else 'Player'


class QAgentAdapter:
    """Adapter to present RLCard state to QLearningAgent in the expected format.
    """
    def __init__(self, qagent):
        self.qagent = qagent
        self.use_raw = True

    def step(self, state):
        # state is RLCard state: dict with 'obs' or 'raw_obs'
        obs_state = self._to_obs_state(state)
        return self.qagent.step(obs_state)

    def eval_step(self, state):
        obs_state = self._to_obs_state(state)
        return self.qagent.eval_step(obs_state)

    def _to_obs_state(self, state):
        # Q-learning agent expects keys 'player_value' and 'dealer_showing'.
        obs = state['obs']
        return {'player_value': int(obs[0]), 'dealer_showing': int(obs[1])}


class MCAgentAdapter:
    """Adapter for MonteCarloBlackjackAgent which expects (player_sum, dealer_showing, usable_ace)
    Uses decode_rlcard_state to convert raw_obs into expected format.
    """
    def __init__(self, mcagent):
        self.mcagent = mcagent
        self.use_raw = True

    def step(self, state):
        # Return action number
        tup = self._to_tuple(state)
        return self.mcagent.get_action(tup[0], tup[1], tup[2])

    def eval_step(self, state):
        tup = self._to_tuple(state)
        a = self.mcagent.get_action(tup[0], tup[1], tup[2])
        return a, {}

    def _to_tuple(self, state):
        if isinstance(state, dict) and 'raw_obs' in state:
            return decode_rlcard_state(state['raw_obs'])
        if isinstance(state, dict) and 'obs' in state:
            obs = state['obs']
            return (int(obs[0]), int(obs[1]), False)
        # Fallback
        return (0, 0, False)


class InteractiveHumanAgent:
    def __init__(self, player_id=0):
        self.player_id = player_id
        self.use_raw = True

    def eval_step(self, state):
        # Show legal actions
        legal = state.get('legal_actions') if isinstance(state, dict) else None
        if isinstance(legal, dict):
            legal_list = list(legal.keys())
        else:
            legal_list = legal or [0, 1]
        action_names = {0: 'Hit', 1: 'Stand'}
        print('Legal actions:', [(a, action_names.get(a, str(a))) for a in legal_list])

        while True:
            try:
                txt = input('Enter action (0=Hit,1=Stand): ').strip()
            except (KeyboardInterrupt, EOFError):
                print('\nExiting.')
                sys.exit(0)
            try:
                a = int(txt)
            except ValueError:
                print('Enter 0 or 1')
                continue
            if legal_list is not None and a not in legal_list:
                print('Action not legal for this state')
                continue
            return a, {}

    def step(self, state):
        """Provided for completeness, not used in practice.
        """
        res = self.eval_step(state)
        # eval_step returns (action, info)
        if isinstance(res, tuple):
            return res[0]
        return res
    

def get_agent_action(agent, state):
    """Get action from agent given RLCard state dict.
    """
    action = None
    res = agent.eval_step(state)
    action = res[0] if isinstance(res, tuple) else res

    # If agent returned (action, info) earlier, ensure action is int
    try:
        action = int(action)
    except Exception:
        print('Warning: could not interpret action from agent; defaulting to Stand (1)')
        action = 1

    return action


def main():
    print('Starting terminal Blackjack: you vs Q-learning vs Monte Carlo agent')
    num_players = 3
    env = rlcard.make('blackjack', config={'game_num_players': num_players})

    # Instantiate agents
    q_agent = QLearningAgent()
    mc_agent = MonteCarloBlackjackAgent()

    # Try to load saved models/strategies so opponents use trained policies if available
    # Both load functions look for default paths in models/ directory
    try:
        q_loaded = q_agent.load_model()
        if q_loaded:
            print('SUCCESS: Loaded Q-learning model from models/q_learning_agent.pkl')
        else:
            print('WARNING: Q-learning model not found — using untrained Q-agent')
    except Exception as e:
        print('WARNING: exception occured loading Q-learning model:', e)

    try:
        mc_loaded = mc_agent.load_strategy()
        if mc_loaded:
            print('SUCCESS: Loaded Monte Carlo strategy from models/monte_carlo_strategy.pkl')
        else:
            print('WARNING: Monte Carlo strategy not found — using untrained Monte Carlo agent')
    except Exception as e:
        print('WARNING: exception occured loading Monte Carlo strategy:', e)

    # Wrap them, to ensure that they function as RLCard agents
    q_adapter = QAgentAdapter(q_agent)
    mc_adapter = MCAgentAdapter(mc_agent)
    human = InteractiveHumanAgent(player_id=0)

    agents = [human, q_adapter, mc_adapter]
    env.set_agents(agents)

    print('Starting game loop.')

    while True:
        sep('New Round')
        # Manual loop so we can show all players' hands as the round progresses
        state, player_id = env.reset()

        # Show initial hands (from raw_obs or state)
        sep('Initial hands')
        for i in range(num_players):
            lbl = f'{agent_label(agents[i], i)} (Player {i})'
            show_hand(i, state, label=lbl)

        # Play until the environment signals the round is over
        while not env.is_over():
            cur_player = env.get_player_id() if hasattr(env, 'get_player_id') else player_id
            # Show current player's visible hand and dealer card
            sep(f"{agent_label(agents[cur_player], cur_player)} turn")
            show_hand(cur_player, state, label=f'{agent_label(agents[cur_player], cur_player)} (Player {cur_player})')
            # Dealer showing
            if isinstance(state, dict):
                ro = state.get('raw_obs', {})
                if isinstance(ro, dict) and 'dealer hand' in ro:
                    dh = ro.get('dealer hand')
                    print('Dealer shows:', dh)
                    print_card(dh)

            agent = agents[cur_player]
            action = get_agent_action(agent, state)
            action_names = {0: 'Hit', 1: 'Stand'}
            print(f'  {agent_label(agents[cur_player], cur_player)} (Player {cur_player}) chooses: {action_names.get(action, action)}')

            # Apply the action
            state, player_id = env.step(action)

            # After the action, show all players' hands
            sep('Hands after action')
            for i in range(num_players):
                lbl = f'{agent_label(agents[i], i)} (Player {i})'
                show_hand(i, state, label=lbl)

        # Show dealer final hand
        sep('Dealer final hand')
        dealer_hand = state.get('raw_obs', {}).get('dealer hand') if isinstance(state, dict) else None
        if dealer_hand is not None:
            if isinstance(dealer_hand, list):
                print_card(dealer_hand)
        else:
            print('  (dealer hand not available)')

        # Show payoffs
        payoffs = env.get_payoffs() if hasattr(env, 'get_payoffs') else []
        print('\nRound finished. Payoffs:')
        for i, p in enumerate(payoffs):
            print(f' {agent_label(agents[i], i)} (Player {i}): {p}')

        # Prompt to play again or quit
        resp = input('\nPress Enter to play another round (q to quit): ').strip().lower()
        if resp in ('q', 'quit', 'exit'):
            print('Quitting...')
            break


if __name__ == '__main__':
    main()
