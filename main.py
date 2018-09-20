import argparse
import math
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="Pendulum-v0",
                    help='name of the environment to run')
parser.add_argument('--deterministic', type=bool, default=False,
                    help='use a deterministic policy (default:False)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluate a policy (default:False)')
parser.add_argument('--reparam', type=bool, default=True,
                    help='reparameterize the policy (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--scale_R', type=int, default=5, metavar='G',
                    help='reward scaling (default: 5)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

# Environment
env = NormalizedActions(gym.make(args.env_name))
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

writer = SummaryWriter()

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
rewards = []
total_numsteps = 0
updates = 0

for i_episode in itertools.count():
    state = env.reset()

    episode_reward = 0
    while True:
        action = agent.select_action(state)  # Sample action from policy
        next_state, reward, done, _ = env.step(action)  # Step
        mask = not done  # 1 for not done and 0 for done
        memory.push(state, action, reward, next_state, mask)  # Append transition to memory
        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step): # Number of updates per step in environment
                # Sample a batch from memory
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.batch_size)
                # Update parameters of all the networks
                value_loss, critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(state_batch, action_batch, 
                                                                                                reward_batch, next_state_batch, 
                                                                                                mask_batch, updates)
                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                updates += 1

        state = next_state
        total_numsteps += 1
        episode_reward += reward

        if done:
            break

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    rewards.append(episode_reward)
    print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, np.round(rewards[-1],2),
                                                                                np.round(np.mean(rewards[-100:]),2)))

env.close()
