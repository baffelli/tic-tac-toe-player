

import numpy as np
import keras
from keras import layers
import tensorflow as tf
from player import make_move, random_move, next_action
import matplotlib.pyplot as plt
from keras import mixed_precision
import board
from player import ModelPlayer, RandomPlayer, AbstractPlayer
from typing import Callable, Optional

from keras import utils

from gymnasium import Env

seed = 42
gamma = 0.95  # Discount factor for past rewards
max_steps_per_episode = 1000
number_episodes = 1000
n_pos = 9
num_symbols = 2

input_shape = (n_pos, )
output_shape = n_pos



def make_model() -> keras.Model:
    input = keras.Input(input_shape)
    dense = layers.Dense(input_shape[0] * 2, activation='relu',
                            kernel_initializer='random_normal')(input)
    dense1 = layers.Dense(18, activation="relu")(input)
    actor = layers.Dense(9, activation='softmax',
                            kernel_initializer='random_normal')(dense)

    critic = layers.Dense(1, kernel_initializer='random_normal')(dense1)
    model = keras.Model(inputs=input, outputs=[
                        actor, critic])
    return model

# Setup model
model = make_model()
utils.plot_model(model, show_shapes=True)
print(model.summary())


optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
black_rewards_history = []
rewards_history = []
running_reward = 0
episode_count = 0
eps = np.finfo(np.float32).eps.item()

winning_history = []




def calculate_running_reward(running, episode, alpha=0.05):
    return alpha * episode + (1 - alpha) * running


def discounted_rewards(rewards_history: np.ndarray, gamma: float) -> list[float]:
    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
    returns = returns.tolist()
    return returns


def play_game(g: board.TicTacToeEnv, player: AbstractPlayer, opponent: AbstractPlayer) -> int:
    state, rest = g.reset()
    if player.color == opponent.color:
        raise ValueError("Opponent and player should have different colors")
    cnt = 0
    while g.board.get_state() == board.BoardState.ONGOING:
        action = player.move(g) if (cnt % 2) == 0 else opponent.move(g)
        state, rewards, done, rest = g.step(action)
        cnt = cnt + 1
    w = g.board.get_winner()
    if w is None:
        return 0
    else:   
        return 1 if w == player.color else -1




#Setup game
starting_color = board.Player.WHITE
game = board.TicTacToeEnv(render_mode="human")
winning_stats = np.zeros(int(number_episodes / 10))
running_reward_history = []





for i in range(number_episodes):  # Run until solved
    state, a = game.reset()
    episode_reward = 0
    #keras.backend.clear_session()
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # of the agent in a pop up window.
            state = game._observation()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])
            # Sample action from action probability distribution
            action = next_action(action_probs)
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            state, reward, done, rest = game.step(action)
            print(action, done)
            rewards_history.append(reward)
            episode_reward += reward
            if done:
                break
        # Update running reward to check condition for solving
        running_reward = calculate_running_reward(running_reward, episode_reward)

        # Calculate expected value from rewards
        returns = discounted_rewards(rewards_history, gamma)

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear() 



    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        #Play test game
        # game.reset()
        model_player = ModelPlayer(starting_color, model)
        random_player = RandomPlayer(starting_color.switch())
        winner = play_game(game, model_player, random_player)
        winning_stats[i // 10] = winner
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))
        current_episodes = winning_stats[0:int(i / 10)]
        winnint_percent = np.sum((current_episodes == 1) + (current_episodes == 0 )) / (current_episodes.shape[0])
        print(winnint_percent)
        starting_color = starting_color.switch()
        running_reward_history.append(running_reward)
    if running_reward > 1e4:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
