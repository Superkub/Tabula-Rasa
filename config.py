# Settings for training and model

MODEL_PATH      = "models/bipedalwalker_ppo"       # Where the final model is saved
BEST_MODEL_PATH = "models/best/best_model"         # Where the best model is saved
LOG_PATH        = "logs/"                          # Where logs are stored

ENV_NAME        = "SisyphusWalker-v0"              # The simulation environment
N_ENVS          = 8                                # Number of parallel environments

TOTAL_TIMESTEPS = 10_000_000                       # Total training steps
LEARNING_RATE   = 0.0003                           # How fast the model learns
N_STEPS         = 4096                             # Steps collected before each update
BATCH_SIZE      = 128                              # Samples per training batch
N_EPOCHS        = 10                               # Times each batch is reused
GAMMA           = 0.99                             # How much future rewards matter
EVAL_FREQ       = 10_000                           # How often the model is evaluated

DEVICE          = "cpu"                            # CPU is faster for small networks like MlpPolicy