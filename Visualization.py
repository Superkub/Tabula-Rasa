import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from config import ENV_NAME, BEST_MODEL_PATH

def run_episodes(model, env, episodes, deterministic=False):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    return rewards

def compare_agents(episodes=3):
    env = gym.make(ENV_NAME)
    untrained = run_episodes(PPO("MlpPolicy", env), env, episodes)
    env.close()
    try:
        env2 = gym.make(ENV_NAME)
        trained = run_episodes(PPO.load(BEST_MODEL_PATH), env2, episodes, deterministic=True)
        env2.close()
    except FileNotFoundError:
        print("⚠️  No trained model found!")
        return
    x = range(1, episodes + 1)
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], untrained, width=0.4, label="Untrained", color="tomato")
    plt.bar([i + 0.2 for i in x], trained,   width=0.4, label="Trained",   color="royalblue")
    plt.title("Tabula Rasa — Untrained vs Trained", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()

def plot_checkpoints():
    d = "models/checkpoints/"
    checkpoints = sorted([f for f in os.listdir(d) if f.endswith(".zip")]) if os.path.exists(d) else []
    if not checkpoints:
        print("No checkpoints found!")
        return
    env, rewards, labels = gym.make(ENV_NAME), [], []
    for ckpt in checkpoints:
        rewards += run_episodes(PPO.load(os.path.join(d, ckpt[:-4])), env, 1, deterministic=True)
        labels.append(ckpt.replace(".zip", "").replace("checkpoint_", ""))
    env.close()
    plt.figure(figsize=(10, 5))
    plt.plot(labels, rewards, marker="o", color="royalblue", linewidth=2)
    plt.title("Tabula Rasa — Checkpoint Progression", fontsize=14)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("checkpoint_progress.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode == "compare":
        compare_agents()
    elif mode == "checkpoints":
        plot_checkpoints()
    else:
        print("Usage: python visualization.py [compare|checkpoints]")