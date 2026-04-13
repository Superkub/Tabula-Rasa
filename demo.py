import time
import numpy as np
from stable_baselines3 import PPO
from config import BEST_MODEL_PATH, DEMO_EPISODES
from environment import make_tabula_rasa_env

def run(model_path=None, episodes=DEMO_EPISODES, max_seconds=None):
    env = make_tabula_rasa_env(render_mode="human")
    model = PPO.load(model_path) if model_path else PPO("MlpPolicy", env)
    episode = 0
    try:
        while True:
            obs, _ = env.reset()
            total, done, start = 0, False, time.time()
            episode += 1
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
                if max_seconds and time.time() - start > max_seconds:
                    break
            print(f"Episode {episode}: Reward = {total:.2f}")
            if not max_seconds and episode >= episodes:
                break
    except KeyboardInterrupt:
        print("\n⏹️  Stopped")
    env.close()

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "trained"
    if mode == "trained":
        run(BEST_MODEL_PATH)
    elif mode == "untrained":
        run()
    elif mode == "checkpoint" and len(sys.argv) > 2:
        run(model_path=sys.argv[2])
    elif mode == "timelapse":
        run(BEST_MODEL_PATH, max_seconds=int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif mode == "timelapse-checkpoint" and len(sys.argv) > 2:
        run(sys.argv[2], max_seconds=int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        print("Usage: python demo.py [trained|untrained|checkpoint <path>|timelapse <seconds>|timelapse-checkpoint <path> <seconds>]")