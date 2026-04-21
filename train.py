# Trains a PPO agent — loads walking model, then learns to push boulder
import os                                          # Create directories
from sisyphus_env import SisyphusEnv               # Sisyphus environment
from stable_baselines3 import PPO                  # PPO algorithm
from stable_baselines3.common.callbacks import (
    CheckpointCallback                             # Saves snapshots of the model at regular intervals
)
from stable_baselines3.common.env_util import make_vec_env  # Runs multiple environments in parallel for faster training
from config import *                               # All settings

def train():
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    print("🖥️  Running on CPU")

    env = make_vec_env(ENV_NAME, n_envs=N_ENVS)
    model = PPO.load("models/bipedalwalker_ppo", env=env,
                     learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE,
                     n_epochs=N_EPOCHS, gamma=GAMMA, device=DEVICE, ent_coef=0.01,
                     verbose=1, tensorboard_log=LOG_PATH)

    callbacks = [
        CheckpointCallback(save_freq=100_000 // N_ENVS, save_path="models/checkpoints/",
                           name_prefix="checkpoint", verbose=1)
    ]

    print("🦾 Training with boulder — Sisyphus mode...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    env.close()

if __name__ == "__main__":
    train()