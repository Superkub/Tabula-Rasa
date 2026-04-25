from stable_baselines3 import PPO
from sisyphus_env import SisyphusEnv

env = SisyphusEnv(render_mode="human")
model = PPO.load("models/checkpoints/checkpoint_20000000_steps", device="cpu")

for ep in range(3):
    obs, _ = env.reset()
    total, done = 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"Episode {ep+1}: Reward = {total:.2f}")
env.close()