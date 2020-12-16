import parl
from parl import layers


class CartpoleModel(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 10

        self.fc1 = layers.fc(size=hid1_size, act="tanh")
        self.fc2 = layers.fc(size=act_dim, act="softmax")  # 执行每个动作的概率

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out
