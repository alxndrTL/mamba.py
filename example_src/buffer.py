import numpy as np

# todo : numpy est une bottleneck ? on a tout qui est en pytorch, on passe par numpy, pour ensuite reconvertir en pytorch pour le training...
# pour la V1 (où la récolte et le training se déroulent pas en mm temps) ce n'est pas très genant

# todo : ce qui est récolté en même temps est fourni en même temps dans les batch (ie across env)... encore une fois, pg pour la V1 les envs sont continuing (cas très particulier...)

class ReplayBuffer():
    def __init__(self, num_envs, capacity, obs_dim, act_dim):
        self.obs_buffer = np.empty((capacity, num_envs, obs_dim), dtype=np.uint8)
        self.act_buffer = np.empty((capacity, num_envs), dtype=np.uint8) # no one-hot
        self.rew_buffer = np.empty((capacity, num_envs), dtype=np.float32)

        self.num_envs = num_envs
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng()

    def store(self, obs, act, rew):
        # obs : (num_envs, L*L)
        # act : (num_envs,)
        # rew : (num_envs,)

        self.obs_buffer[self.idx] = obs
        self.act_buffer[self.idx] = act
        self.rew_buffer[self.idx] = rew

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, batch_len):
        assert self.size >= batch_len, "not enough experience stored"

        start_idxs = self.rng.integers(0, self.size - batch_len, size=batch_size) # (B,)
        env_idxs = self.rng.integers(0, self.num_envs, size=batch_size)[:, None] # (B, 1)

        # all indices for sampling : from start_idxs to start_idxs+batch_len
        idxs = start_idxs[:, None] + np.arange(batch_len) # (B, batch_len)

        batch_obs = self.obs_buffer[idxs, env_idxs]
        batch_acts = self.act_buffer[idxs, env_idxs]
        batch_rews = self.rew_buffer[idxs, env_idxs]
        
        batch = {'obs': batch_obs, 'act': batch_acts, 'rew': batch_rews}
        return batch
    