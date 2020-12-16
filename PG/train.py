"""
policy gradient
"""
import gym
from parl.utils import logger
from .model import CartpoleModel
from .algo import PolicyGradient
from .agent import Agent

ENV_SEED = 1


def main():
    env = gym.make('CartPole-v0')
    action_dim = env.action_space.n
    obs_shape = env.observation_space.shape[0]
    logger.info('obs_dim {}, act_dim {}'.format(obs_shape, action_dim))

    model = CartpoleModel(act_dim=action_dim)
    algorithm = PolicyGradient(model=model, lr=5 * 1e-4)
    agent = Agent(algorithm=algorithm, obs_dim=obs_shape, act_dim=action_dim)


    # while len(rpm) < MEMORY_WARMUP_SIZE:  # warm up replay memory
    #     run_episode(agent, env, rpm)
    #
    # max_episode = 2000
    #
    # # start train
    # episode = 0
    # while episode < max_episode:
    #     # train part
    #     for i in range(0, 50):
    #         total_reward = run_episode(agent, env, rpm)
    #         episode += 1
    #
    #     eval_reward = evaluate(agent, env)
    #     logger.info('episode:{}    test_reward:{}'.format(
    #         episode, eval_reward))


if __name__ == '__main__':
    main()
