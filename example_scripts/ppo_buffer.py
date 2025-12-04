import numpy as np
import torch


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.9, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        """
        Add one timestep of agent-env interaction
        """
        assert self.ptr < self.max_size, "Buffer overflow!"

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = rew
        self.ptr += 1

    def finish_path(self, last_value=0):
        """
        Compute GAE advantage and rewards-to-go
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Normalize advantages and return tensors
        """
        assert self.ptr == self.max_size

        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        return (
            torch.tensor(self.obs_buf, dtype=torch.float32),
            torch.tensor(self.act_buf, dtype=torch.float32),
            torch.tensor(self.adv_buf, dtype=torch.float32),
            torch.tensor(self.ret_buf, dtype=torch.float32),
            torch.tensor(self.logp_buf, dtype=torch.float32),
        )

    @staticmethod
    def discount_cumsum(x, discount):
        """
        Return discounted cumulative sum of vectors
        """

        return np.array(
            [np.sum(x[i:] * discount ** np.arange(len(x) - i)) for i in range(len(x))]
        )
