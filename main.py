import argparse
import math
import gym
import numpy as np
from gym import wrappers
import torch
from sac import SAC
from plot import plot_line
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--algo', default='SAC(GMM)',
                    help='algorithm to use: SAC | SAC(GMM)')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--reparam', type=bool, default=True,
                    help='reparameterize the policy (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--k', type=int, default=4, metavar='G',
                    help='No. of Mixtures (default: 4)')
parser.add_argument('--scale_R', type=int, default=5, metavar='G',
                    help='reward scaling (default: 5)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

env = NormalizedActions(gym.make(args.env_name))

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
agent = SAC(env.observation_space.shape[0], env.action_space, args)


memory = ReplayMemory(args.replay_size)


rewards = []
total_numsteps = 0
updates = 0

for i_episode in range(args.num_episodes):
    state = env.reset()

    episode_reward = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        mask = not done

        memory.push(state, action, reward, next_state, mask)
        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.batch_size)
                agent.update_parameters(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, total_numsteps)

        state = next_state
        total_numsteps += 1
        episode_reward += reward

        if done:
            break

    rewards.append(episode_reward)
    plot_line(total_numsteps, rewards, args.algo)
    print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, np.round(rewards[-1],2),
                                                                                np.round(np.mean(rewards[-100:]),2)))

env.close()
