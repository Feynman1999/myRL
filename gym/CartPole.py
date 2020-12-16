import gym
import time

env = gym.make("CartPole-v0")
env.reset()
print(env.action_space)
for _ in range(10):
    x = env.step(env.action_space.sample())
    print(x)

env.close()