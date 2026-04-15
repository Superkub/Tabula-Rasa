# Shows the agent visually — trained or untrained
from stable_baselines3 import PPO                  # PPO algorithm
from config import BEST_MODEL_PATH, DEVICE         # Settings
from sisyphus_env import SisyphusEnv               # Sisyphus environment class

def run(model_path=None, episodes=3):
    env = SisyphusEnv(render_mode="human")         # Open visual window directly
    if model_path:
        model = PPO.load(model_path, device=DEVICE)  # Load trained model
    else:
        model = PPO("MlpPolicy", env, device=DEVICE) # Create new untrained model
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
        run()                                      # Random agent — no training
    else:
        run(BEST_MODEL_PATH)                       # Best trained agent