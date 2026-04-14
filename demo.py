# Shows the agent visually - trained or untrained

import time       #track episode duration
import gymnasium as gym #Simulation environment
from stable_baselines3 import PPO #PPO algorithm
from config import ENV_NAME, BEST_MODEL_PATH #Settings

def run(model_path=None, episodes=3):
    env = gym.make(ENV_NAME, render_mode="human") #Open visual window
    model = PPO.load(model_path) if model_path else PPO("MlpPolicy", env) #Load trained or create new
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        print(f"Episode {ep}: Reward = {total:.2f}")
    env.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "untrained":
        run() #random agent - no training
    else:
        run(BEST_MODEL_PATH) # Best trained agent