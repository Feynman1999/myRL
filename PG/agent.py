import parl
from paddle import fluid
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = fluid.layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = fluid.layers.data(name="obs", shape=[self.obs_dim], dtype='float32')
            act = fluid.layers.data(name="act", shape=[1], dtype='int64')
            reward = fluid.layers.data(name="reward", shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # [4] -> [1,4]
        act_prob = self.fluid_executor.run(self.pred_program, feed={'obs': obs.astype('float32')},
                                           fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.random.choice(range(self.act_dim), p=act_prob)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)  # [4] -> [1,4]
        act_prob = self.fluid_executor.run(self.pred_program, feed={'obs': obs.astype('float32')},
                                           fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost
