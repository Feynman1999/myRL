import parl
from parl import layers
from paddle import fluid


class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr=None):
        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, action, reward):
        """

        :param obs:      [B,4]
        :param action:   [B,1]
        :param reward:   [B,]
        :return:
        """
        act_prob = self.model(obs)  # [B,2]
        # [B, 2] -> [B, ]
        log_prob = layers.reduce_sum(-1.0 * layers.log(act_prob) * layers.one_hot(action, depth=act_prob.shape[1]), dim=1, keep_dim=False)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)

        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost
