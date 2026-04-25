# Technical Documentation — Tabula Rasa

## 1. Project Overview

Tabula Rasa uses PPO (Proximal Policy Optimization) to train a bipedal agent to walk and push a boulder up a slope — inspired by the myth of Sisyphus. The project uses transfer learning across three training phases, where each phase loads the model from the previous phase and trains further.

## 2. Data Flow Diagram

The program is split into separate files that each do one job. Settings start at config.py, get used by train.py to train the agent inside the custom environment sisyphus_env.py, and the trained model is saved to disk. demo.py loads the model and visualizes the agent.

┌──────────┐
│   User   │──── python train.py ───────────────────┐
│          │──── python demo.py ─────────────────────┼──┐
└──────────┘                                        │  │
▼  │
┌──────────────┐   settings   ┌──────────────────┐  │  │
│  config.py   │─────────────>│    train.py      │<─┘  │
│              │              │                  │     │
│ Hyperparams, │   settings   │ PPO training,    │     │
│ paths,       │──────┐       │ transfer learning│     │
│ device       │      │       └──────┬───────────┘     │
└──────────────┘      │              │  ▲              │
│     actions (4) │  │ obs (24),  │
│              │  │ reward       │
│              ▼  │              │
┌──────────────┐      │       ┌──────────────────┐     │
│BipedalWalker │ extends      │ sisyphus_env.py  │     │
│    -v3       │─────────────>│                  │     │
│              │              │ Boulder physics, │     │
│ Gymnasium /  │              │ slope terrain,   │     │
│ Box2D        │              │ reward system    │     │
└──────────────┘              └──────────────────┘     │
│
┌─────────────────┐  save .zip  ┌─────────────┐   │
│     logs/       │<────────────│             │   │
│                 │             │  train.py   │   │
│ TensorBoard     │  save .zip  │             │   │
│ training data   │   ┌────────>│             │   │
└─────────────────┘   │        └─────────────┘   │
│                           │
┌─────────────────┐   │ transfer  ┌─────────────┐ │
│    models/      │───┘ learning  │   demo.py   │<┘
│                 │──────────────>│             │
│ bipedalwalker   │  load model   │ Visual      │
│ _ppo.zip        │              │ playback    │──> Pygame window
│ + checkpoints   │              └─────────────┘
└─────────────────┘

| Stage | Input | Output |
|-------|-------|--------|
| config.py | None | Settings: learning rate, timesteps, batch size, device, paths |
| train.py | Settings from config, model from models/, environment from sisyphus_env | Trained model (.zip), checkpoints, TensorBoard logs |
| sisyphus_env.py | Actions (4 motor torques) from train.py or demo.py | Observations (24 values), reward, boulder/slope physics |
| demo.py | Settings from config, trained model from models/ | Visual rendering in Pygame window |

## 3. Component Descriptions

### 3.1 config.py

Stores all settings and hyperparameters in one place. Every other file imports from here so values only need to be changed once.

Contains: model save paths, environment name (SisyphusWalker-v0), number of parallel environments (32), total timesteps (30M), learning rate (0.0003), batch size (128), gamma (0.99), and device (CPU).

### 3.2 train.py

The training script. Loads the previous model from models/bipedalwalker_ppo.zip (transfer learning) and trains it further using PPO from Stable-Baselines3. Runs 32 parallel environments for faster data collection.

Uses CheckpointCallback to save model snapshots every 100K steps. The entropy coefficient (ent_coef=0.01) was added to prevent the agent from stopping exploration — without it, the agent's standard deviation collapsed to 0.02 and it stopped trying new strategies.

The training happened in three phases:
- Phase 1: BipedalWalker-v3, 10M steps — learned to walk (reward ~280)
- Phase 2: SisyphusWalker-v0 with boulder, 20M steps — learned to push (reward ~320)
- Phase 3: SisyphusWalker-v0 with boulder + slope, 30M steps — learned to push uphill

### 3.3 sisyphus_env.py

A custom Gymnasium environment that extends BipedalWalker-v3. Adds two physical objects and a reward system:

**Boulder:** An 8-sided polygon (Box2D dynamic body) placed 20 units ahead of the agent. Uses irregular vertex offsets for a natural rock appearance. Properties: density 0.5, friction 0.8, restitution 0.05. Rendered via BipedalWalker's built-in drawlist with color1=(120,110,90) and color2=(80,70,55).

**Slope:** A mountain profile generated with an exponential curve (t²) combined with sine/cosine waves for rocky texture. Gets steeper toward the top. Physics uses edge fixtures (friction 2.0) for collision. Visuals use polygon segments registered as isSensor fixtures in the drawlist. Length: 30 units, height: 10 units.

**Reward system:** Two components on top of BipedalWalker's base reward:
1. Progress reward — boulder movement × 2 (agent earns points when boulder moves forward)
2. Proximity bonus — +0.1 when agent is within 2 units of boulder

The environment registers itself as "SisyphusWalker-v0" via gym.register so train.py can find it through make_vec_env.

### 3.4 demo.py

Visual playback script. Creates a SisyphusEnv directly (not through gym.make, which wraps the environment in layers that bypass custom rendering). Loads the trained model and runs episodes visually.

Two modes:
- `python demo.py` — loads the trained model and shows the agent walking, pushing the boulder, and climbing the slope
- `python demo.py untrained` — creates a random PPO model with no training, showing the agent failing

## 4. Training Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| total_timesteps | 60 000 000 (across all three transfer learning phases) | Enough for agent to learn walking + pushing + climbing |
| n_envs | 32 | Parallel environments for faster data collection |
| learning_rate | 0.0003 | Small stable steps — standard for PPO |
| n_steps | 4096 | Steps collected before each policy update |
| batch_size | 128 | Good balance for MlpPolicy (2×64 neurons) |
| n_epochs | 10 | Times each batch is reused for training |
| gamma | 0.99 | Values future rewards almost as much as immediate ones |
| ent_coef | 0.01 | Forces continued exploration — prevents entropy collapse |
| device | CPU | Faster than GPU for small networks like MlpPolicy |

## 5. Technical Challenges

**Rendering the boulder:** gym.make() wraps the environment in layers that bypassed our custom step() method. As a result, the boulder existed physically but was not rendered. This was resolved by instantiating SisyphusEnv directly in demo.py.

**Double observation bug:** During early testing with a 28-dimensional observation space, super().reset() internally called step(), causing boulder observations to be appended twice (32 instead of 28 values). This was initially mitigated with a length check before appending.

**Observation space mismatch:** A pre-trained walking model (24 inputs) could not be loaded into a model expecting 28 inputs. This exposed a compatibility constraint between architectures and led to a redesign of the observation space.

**Observation design decision:** The final system maintains a 24-dimensional observation space. Although explicit boulder coordinates were initially considered, the agent proved capable of inferring the boulder’s presence through proprioception (e.g., joint resistance and velocity changes). Retaining the standard 24 inputs ensured full compatibility with pre-trained models while encouraging learning through physical interaction rather than explicit state encoding.

**Reward balancing:** Initially, the boulder progress reward was set to a multiplier of 5. This caused the agent to over-prioritize the boulder, leading to unstable movements. By reducing this to 2, we forced the agent to maintain its core walking gait while still incentivizing upward progress.

**Entropy collapse:** Early training caused std to collapse to 0.02 — the agent stopped exploring. Fixed by adding ent_coef=0.01 to force continued exploration.

**Rendering flicker:** Double pygame.display.flip() caused screen blinking. Solved by using BipedalWalker's built-in drawlist for rendering boulder and slope instead of manual Pygame drawing.

**Boulder Density Adjustment:** > During Phase 2 (Flat ground), the boulder density was set to 1.5. However, when moving to Phase 3 (The Slope), this weight proved too high for the walker’s torque limits, causing it to fall backward. We tuned the density down to 0.5, balancing the physical challenge with the agent's ability to maintain balance while exerting upward force.

**Surface friction tuning:** The default friction (0.8) was insufficient for the steep slope, causing the agent’s feet to slip and the boulder to slide back. We increased the slope friction to 2.0, providing necessary traction for the agent to exert force effectively.

## 6. How to Run

Requirements: Python 3.12, gymnasium[box2d], stable-baselines3, tensorboard, torch

Installation:
pip install -r requirements.txt
Running:
python train.py                                # Train (loads previous model, continues training)
python demo.py                                 # Watch trained agent with boulder and slope
python demo.py untrained                       # Watch untrained agent falling
## 7. Project Structure
Tabula-Rasa
├── config.py                                  # Settings and hyperparameters
├── train.py                                   # Training with transfer learning and checkpoints
├── demo.py                                    # Watch agent visually (trained or untrained)
├── sisyphus_env.py                            # Custom environment with boulder and slope
├── test_bipedalwalker10m.py                   # Test Phase 1 — walking
├── test_sisyphuswalkerwithboulder20m.py       # Test Phase 2 — boulder (slope visible but untrained)
├── test_sisyphuswalkerwithboulderslope30m.py  # Test Phase 3 — slope
├── requirements.txt                           # Python package dependencies
├── readme.md                                  # Project overview and usage
├── TECHNICAL_DOCUMENTATION.md                 # This file
├── models/
│   ├── bipedalwalker_ppo.zip                  # Final trained model
│   └── checkpoints/                           # Model snapshots during training
└── logs/                                      # TensorBoard training logs

## 8. Stack

- Farama Gymnasium — BipedalWalker-v3
- Stable-Baselines3 — PPO
- PyTorch (backend)
- Box2D (physics)
