"""
Watch the AI drive (or fail to drive).
"""
import gymnasium as gym
from stable_baselines3 import PPO

# render_mode="human" opens a window we can watch.
env = gym.make("CarRacing-v3", render_mode="human")
model = PPO.load("practice_racer", device="cuda")

print("Loading the AI's brain...")
print("Press Ctrl+C in this terminal to stop.")
print()

obs, _ = env.reset()
total_reward = 0
episode = 1

for step in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode {episode} done. Total score: {total_reward:.1f}")
        episode += 1
        total_reward = 0
        obs, _ = env.reset()

env.close()