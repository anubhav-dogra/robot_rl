from ppo_buffer import PPOBuffer

buf = PPOBuffer(obs_dim=11, act_dim=2, size=5)


for _ in range(5):
    buf.store(obs=[0] * 11, act=[0, 0], rew=1.0, val=0.5, logp=-1.0)

buf.finish_path(last_value=0)

data = buf.get()
for d in data:
    print(d.shape)
