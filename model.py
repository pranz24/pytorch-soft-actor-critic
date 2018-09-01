import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import create_log_gaussian, logsumexp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x1 = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.cat([state, action], 1)
        x2 = F.relu(self.linear4(x2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2
    
    
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def evaluate(self, state, reparam=False, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)

        if reparam == True:
            x_t = normal.rsample() #mean + std * torch.randn(1,6)
        else:
            x_t = normal.sample()

        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, x_t, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        _, _, x_t, _, _ = self.evaluate(state)
        action = torch.tanh(x_t)
        action = action.detach().cpu().numpy()
        return action[0]


class GaussianMixturePolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, k):
        super(GaussianMixturePolicy, self).__init__()
        self.actions = num_actions
        self.k = k

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.out_linear = nn.Linear(hidden_size, (k * 2 * self.actions) + k)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        out = self.out_linear(x)
        out = out.view(-1, self.k, (2 * self.actions) + 1)
        log_w = out[:, :, 0]
        mean = out[:, :, 1:1 + self.actions]
        log_std = torch.clamp(out[:, :, 1 + self.actions:], min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return log_w, mean, log_std

    def evaluate(self, state, reparam=False, epsilon=1e-6):
        log_w, mean, log_std = self.forward(state)
        std = log_std.exp()
        W = F.softmax(log_w, dim=1)
        pi_picked = torch.multinomial(W, num_samples=1)
        for i, r in enumerate(pi_picked):
            means = mean[:, r, :]
            means = means[:, 0, :]
            stds = std[:, r, :]
            stds = stds[:, 0, :]


        # We can only reparameterize if there was one component in the GMM,
        # in which case one should use GaussianPolicy
        normal = Normal(means, stds)
        x_t = normal.sample()
        action = torch.tanh(x_t)

        log_prob = create_log_gaussian(mean, log_std, x_t[:, None, :]) - torch.log(1 - action.pow(2) + epsilon).sum(
            dim=-1, keepdim=True)
        log_prob = logsumexp(log_prob + log_w, dim=-1, keepdim=True)
        log_prob = log_prob - logsumexp(log_w, dim=-1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        _, _, x_t, _, _ = self.evaluate(state)
        action = torch.tanh(x_t)

        action = action.detach().cpu().numpy()
        return action[0]
