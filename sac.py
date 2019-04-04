import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, ValueNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.num_inputs = num_inputs
        self.max_action = float(action_space.high[0])
        self.action_space = action_space.shape[0]
        self.gamma = args.gamma
        self.tau = args.tau

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.critic = QNetwork(self.num_inputs, self.action_space, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        if self.policy_type == "Gaussian":
            self.alpha = args.alpha
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)


            self.policy = GaussianPolicy(self.num_inputs, self.action_space, args.hidden_size, self.max_action).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            self.value = ValueNetwork(self.num_inputs, args.hidden_size).to(self.device)
            self.value_target = ValueNetwork(self.num_inputs, args.hidden_size).to(self.device)
            self.value_optim = Adam(self.value.parameters(), lr=args.lr)
            hard_update(self.value_target, self.value)
        else:
            self.policy = DeterministicPolicy(self.num_inputs, self.action_space, args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            self.critic_target = QNetwork(self.num_inputs, self.action_space, args.hidden_size).to(self.device)
            hard_update(self.critic_target, self.critic)



    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            self.policy.train()
            action, _, _ = self.policy.sample(state)
        else:
            self.policy.eval()
            _, _, action = self.policy.sample(state)
        action = action.detach().cpu().numpy()
        return action[0]



    def update_parameters(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, updates):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        qf1, qf2 = self.critic(state_batch, action_batch) # Two Q-functions to mitigate positive bias in the policy improvement step
        pi, log_pi, _ = self.policy.sample(state_batch)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
                alpha_logs = torch.tensor(self.alpha) # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_logs = torch.tensor(self.alpha) # For TensorboardX logs

            vf = self.value(state_batch) # separate function approximator for the soft value can stabilize training.
            with torch.no_grad():
                vf_next_target = self.value_target(next_state_batch)
                next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_logs = self.alpha  # For TensorboardX logs
            with torch.no_grad():
                next_state_action, _, _, _, _, = self.policy.sample(next_state_batch)
                # Use a target critic network for deterministic policy and eradicate the value value network completely.
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if self.policy_type == "Gaussian":
            vf_target = min_qf_pi - (self.alpha * log_pi)
            value_loss = F.mse_loss(vf, vf_target.detach()) # JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - α * log π(at|st)]))^2]

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # Regularization Loss
        # mean_loss = 0.001 * mean.pow(2).mean()
        # std_loss = 0.001 * log_std.pow(2).mean()

        # policy_loss += mean_loss + std_loss

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        if self.policy_type == "Gaussian":
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
        else:
            value_loss = torch.tensor(0.).to(self.device)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        
        """
        We update the target weights to match the current value function weights periodically
        Update target parameter after every n(args.target_update_interval) updates
        """
        if updates % self.target_update_interval == 0 and self.policy_type == "Deterministic":
            soft_update(self.critic_target, self.critic, self.tau)

        elif updates % self.target_update_interval == 0 and self.policy_type == "Gaussian":
            soft_update(self.value_target, self.value, self.tau)
        return value_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_logs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        if value_path is None:
            value_path = "models/sac_value_{}_{}".format(env_name, suffix)
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
        torch.save(self.value.state_dict(), value_path)
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path, value_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state_dict(torch.load(value_path))
