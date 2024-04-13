import gymnasium as gym
import numpy as np

env = gym.make("Ant-v4", render_mode="human")
observation, info = env.reset()
action = env.action_space.sample()  # agent policy that uses the observation and info

"""
Actions (Torques -1 - 1 N m):
0 - Back Right Hip
1 - Back Right Knee
2 - Back Left Hip
3 - Back Left Knee
4 - Front Right Hip
5 - Front Right Knee
6 - Front Left Hip
7 - Front Left Knee
"""
zeros = np.zeros_like(action)
for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
