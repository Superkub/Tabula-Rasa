# Settings for training and model
import torch #Check if GPU is available

MODEL_PATH      = "models/bipedalwalker_ppo" #where the final model is saved
BEST_MODEL_PATH = "models/best/best_model" #where the best model is saved
LOG_PATH        = "logs/" #where logs are stored

ENV_NAME        = "BipedalWalker-v3" #The simulation environment
N_ENVS          = 8 #number of parallel environments

TOTAL_TIMESTEPS = 10_000_000 #total training steps
LEARNING_RATE   = 0.0003 #how fast the model learns
N_STEPS         = 4096 #steps collected before each update
BATCH_SIZE      = 128 #samples per training batch
N_EPOCHS        = 10 #times each batch is reused
GAMMA           = 0.99 #how much future rewards matter
EVAL_FREQ       = 10_000 #how often the model is evaluated

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu" #Choose GPU if available, otherwise go for CPU