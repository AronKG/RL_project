import rlcard
from rlcard.agents import DQNAgent

env = rlcard.make('blackjack')
print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)

agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64, 64]
)

env.set_agents([agent])

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

with Logger("experiments/blackjack_dqn_result") as logger:
    for episode in range(4):

        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        print(f'Episode {episode}: reward {payoffs[0]}')
        print("Trajectories (Before reorganize):")
        print(trajectories)

        # Reorganize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)
        print("Trajectories (After reorganize):")
        print(trajectories)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)

        # Evaluate the performance
        if episode % 2 == 0:
            logger.log_performance(
                env.timestep,
                tournament(
                    env,
                    2,
                )[0]
            )

    # Get the paths
    csv_path, fig_path = logger.csv_path, logger.fig_path

plot_curve(csv_path, fig_path, "DQN")