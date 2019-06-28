import sys
import gym
import pylab
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical

EPISODES = 1000

'''
Set up code taken from repo: https://github.com/rlcode/
This class uses the categorical crossentropy loss function where the true predictions represent the one-hot of the
actions taken by the RL agent during the episode. It uses a custom loss function to scale this using the rewards
obtained in that episode.
'''


def get_discounted_rewards(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 2, 3], 0.99) -> [5.92, 4.97, 3]
    """
    r.reverse()
    prior = 0
    out = []
    for val in r:
        new_val = val + prior * gamma
        out.append(new_val)
        prior = new_val
    out = np.array(out[::-1])
    return out.reshape(out.shape[0], 1)


def run(render):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Make policy gradient RL agent
    state_input = Input(shape=(state_size,), name='state_input')
    dis_rewards = Input(shape=(None,), name='reward_input')
    x = Dense(24, activation='relu')(state_input)
    x = Dense(24, activation='relu')(x)
    y = Dense(action_size, activation='softmax')(x)
    RL_agent_train = Model([state_input, dis_rewards], y)
    RL_agent_predict = Model(state_input, y)

    def policy_gradient_loss(y_true, y_pred):
        # y_true has the index of the action taken by the RL agent
        return categorical_crossentropy(y_true, y_pred) * dis_rewards

    optimizer = Adam(lr=1e-2)
    RL_agent_train.compile(loss=policy_gradient_loss, optimizer=optimizer)

    scores, episodes = [], []
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        states = None
        actions = None
        rewards = []

        while not done:
            # If GUI is desired, give it to them
            # Takes more time
            if render:
                env.render()

            # Determine action that needs to be done
            pi = RL_agent_predict.predict(state)
            action = np.random.choice(range(2), p=pi[0])
            next_state, reward, done, info = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100

            # --- Save s ---
            if states is None:
                states = state
            else:
                states = np.vstack([states, state])

            # --- Save a ---
            action = to_categorical(action, num_classes=action_size, dtype='float32')
            action = np.expand_dims(action, axis=0)
            if actions is None:
                actions = action
            else:
                actions = np.vstack([actions, action])

            # --- Save r ---
            rewards.append(reward)
            score += reward
            state = next_state

            if done:
                # Train model on the current episode's data
                discounted_rewards = get_discounted_rewards(rewards, 1.0)
                model_loss = RL_agent_train.train_on_batch([states, discounted_rewards], actions)

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./cartpole_reinforce.png")
                print("Episode: {} \tScore: {} \tLoss: {}".format(e, score, model_loss))

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(20, len(scores)):]) > 490:
                    sys.exit()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action="store_true",
                        help='If provided, renders GUI')
    args = parser.parse_args()
    run(args.render)
