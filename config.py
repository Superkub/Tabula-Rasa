import torch

MODEL_PATH      = "models/bipedalwalker_ppo"
BEST_MODEL_PATH = "models/best/best_model"
LOG_PATH        = "logs/"

ENV_NAME        = "BipedalWalker-v3"
N_ENVS          = 8

TOTAL_TIMESTEPS = 10_000_000
LEARNING_RATE   = 3e-4
N_STEPS         = 4096
BATCH_SIZE      = 128
N_EPOCHS        = 10
GAMMA           = 0.99
EVAL_FREQ       = 10_000

DEMO_EPISODES   = 5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"