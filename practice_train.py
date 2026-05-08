"""
Mikey's first AI: teaching a tiny car to drive a 2D race track.
This is practice for the BeamNG project.
"""
import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment. CarRacing-v3 is a top-down 2D race track.
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Create the AI brain. It uses a Convolutional Neural Network (CNN)
# because the input is pictures of the track.
# device="cuda" tells it to use the RTX 4070, not the CPU.
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda",
)

# Train for 100,000 steps. This takes about 20-30 minutes on the 4070.
print("Training started. Open a second terminal and run:")
print("    tensorboard --logdir ./logs/")
print("Then open http://localhost:6006 in a browser to watch it learn.")
print()

model.learn(total_timesteps=100_000)

# Save the trained brain so we can use it later.
model.save("practice_racer")
print("Done! Model saved as practice_racer.zip")