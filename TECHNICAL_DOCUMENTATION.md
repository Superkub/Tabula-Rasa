# Technical Documentation — Tabula Rasa

## Overview
Tabula Rasa uses PPO to train a bipedal agent in BipedalWalker-v3 from scratch through trial and error.

## Environment — BipedalWalker-v3
- **Observations:** 24 values (joint angles, velocities, contact sensors)
- **Actions:** 4 continuous motor torque values
- **Reward:** Positive for forward movement, negative for falling
- **Solved:** Average reward > 300

## Algorithm — PPO
1. Collect experience from environment
2. Calculate policy update
3. Clip update to avoid large changes
4. Repeat

## Neural Network — MlpPolicy
- Two hidden layers, 64 neurons each, Tanh activation
- Runs on GPU (CUDA) if available


## Data Flow
[BipedalWalker-v3] → observations (24) → [PPO Policy Network]
↓
reward + observation ← [Environment] ← actions (4)
↓
[Rollout Buffer] → [PPO Update] → [Save Model]


## Training Parameters
| Parameter       | Value      |
|-----------------|------------|
| total_timesteps | 10 000 000 |
| n_envs          | 8          |
| learning_rate   | 0.0003     |
| n_steps         | 4096       |
| batch_size      | 128        |
| n_epochs        | 10         |
| gamma           | 0.99       |

## Files
- `config.py` — Settings
- `train.py` — Training + checkpoints
- `demo.py` — Watch agent

## GitHub
[Link to repository]