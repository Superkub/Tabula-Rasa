# Trains a PPO agent from scratch
import os                                          # Create directories
import gymnasium as gym                            # Simulation environment
import sisyphus_env                                # Registers the Sisyphus environment
from stable_baselines3 import PPO                  # PPO algorithm
from stable_baselines3.common.callbacks import (
    EvalCallback,                                  # Evaluates the model during training and saves the best one
    CheckpointCallback                             # Saves snapshots of the model at regular intervals
)
from stable_baselines3.common.env_util import make_vec_env  # Runs multiple environments in parallel for faster training
from config import *                               # All settings

def train():
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    print("🖥️  Running on CPU")

    env = make_vec_env(ENV_NAME, n_envs=N_ENVS)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_PATH,
                learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS, gamma=GAMMA, device=DEVICE)

    callbacks = [
        EvalCallback(gym.make(ENV_NAME), best_model_save_path="models/best/",
                     log_path=LOG_PATH, eval_freq=EVAL_FREQ, verbose=1),
        CheckpointCallback(save_freq=100_000 // N_ENVS, save_path="models/checkpoints/",
                           name_prefix="checkpoint", verbose=1)
    ]

    print("🦾 Starting training — tabula rasa...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    env.close()

if __name__ == "__main__":
    train()