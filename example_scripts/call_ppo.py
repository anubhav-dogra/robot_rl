import torch
from ppo_policy import PPOPolicy

policy = PPOPolicy(obs_dim=11, act_dim=2)
obs = torch.randn(1, 11)
mean, value = policy(obs)
print(mean, value)
