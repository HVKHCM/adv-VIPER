import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar
import gym
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam 
from gym.envs.toy_text.taxi import *

env_taxi = gym.make("Taxi-v3", render_mode='rgb_array').env
#env.render() 

class taxi:
    def __init__(self, env_taxi, optimizer):
        # Initialize attributes
        self._state_size = env_taxi.observation_space.n
        self._action_size = env_taxi.action_space.n
        self._optimizer = optimizer
        self.expirience_replay_memory = deque(maxlen=2000)
         # Initialize discount and exploration rate
        self.discount = 0.6
        self.exploration = 0.1
         # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_both_model()
    def gather(self, state, action, reward, next_state, terminated):
        self.expirience_replay_memory.append((state, action, reward, next_state, terminated))
    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model
    def align_both_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    def active(self, state):
        if np.random.rand() <= self.exploration:
            return env_taxi.action_space.sample()
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])
    def predict(self, state):
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])
    def predict_q(self, state):
        return self.q_network.predict(state)
    def retraining(self, batch_size):
        minbatch = random.sample(self.expirience_replay_memory, batch_size)
        for state, action, reward, next_state, terminated in minbatch:
            target = self.q_network.predict(state)
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.discount * np.amax(t)
            self.q_network.fit(state, target, epochs=1, verbose=0)


def train():

#ENV_NAME = "Taxi-v3"
#env = gym.make(ENV_NAME)
#env.render()

    optimizer = Adam(learning_rate=0.01)
    agent = taxi(env_taxi, optimizer)
    batch_size = 32
    num_of_episodes = 2
    timesteps_per_episode = 10
    agent.q_network.summary()

    for e in range(0, num_of_episodes):
        print("Episode {}".format(e))
    # Reset the environment
        state, info = env_taxi.reset()
    #print(state)
    #print(info)
        state = np.reshape(state, [1, 1])

    # Initialize variables
        reward = 0
        terminated = False
    #bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
        for timestep in range(timesteps_per_episode):
        # Run Action
            action = agent.active(state)
        # Take action    
            next_state, reward, terminated, info, _ = env_taxi.step(action) 
            next_state = np.reshape(next_state, [1, 1])
            agent.gather(state, action, reward, next_state, terminated)
            state = next_state
            if terminated:
                agent.align_both_model()
                break
            if len(agent.expirience_replay_memory) > batch_size:
                agent.retraining(batch_size)
        #if timestep%10 == 0:
        #    bar.update(timestep/10 + 1)
    #bar.finish()
        if (e + 1) % 10 == 0:
            print("**********************************")
            print("Episode: {}".format(e + 1))
            env_taxi.render()
            print("**********************************")
    return agent

def test(policy, env, render):
    env_decode = TaxiEnv()
    obs, info = env.reset()
    row, column, passe, des = env_decode.decode(obs)
    print(row, column, passe, des)
    obs = np.reshape(obs, [1, 1])
    print(obs)
    rollout = []
    act = policy.predict_act(obs)
    print(act)

        # Step
    next_obs, rew, done, info, _ = env.step(act)

        # Rollout (s, a, r)
    rollout.append((obs, act, rew))

        # Update (and remove LazyFrames)
    obs = np.array(next_obs)
       

    return rollout


#taxi = train()
#taxi.q_network.save('taxi.keras')