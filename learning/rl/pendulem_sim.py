import gymnasium as gym

env = gym.make("Pendulum-v1", g=9.81)

observation, info = env.reset(seed=42)
for _ in range(200):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
env.close()
