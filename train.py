import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from config import ENV_NAME, MODEL_PATH, LOG_PATH, N_ENVS, TOTAL_TIMESTEPS, LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS, GAMMA, EVAL_FREQ, DEVICE
from environment import make_tabula_rasa_env

def train():
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    env = make_vec_env(lambda: make_tabula_rasa_env(render_mode=None), n_envs=N_ENVS)
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