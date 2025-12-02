import gymnasium as gym

env = gym.make("Reacher-v5")
obs, info = env.reset(seed=0)
print("Observation shape:", env.observation_space.shape)
print("Action space:", env.action_space)
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"step {i}: reward={reward}, terminated={terminated}, truncated={truncated}"
    )
env.close()
