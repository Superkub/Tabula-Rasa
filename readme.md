# Tabula Rasa 🦾
> *"One must imagine Sisyphus happy"* — Albert Camus

**GitHub Repository:** https://github.com/Superkub/Tabula-Rasa

A reinforcement learning project where a bipedal agent learns to walk, push a boulder, and climb a slope — from scratch. Inspired by Camus and the myth of Sisyphus. Built with transfer learning across three training phases.

**Group 2 — ML2**

## Stack
- Farama Gymnasium — BipedalWalker-v3
- Stable-Baselines3 — PPO
- PyTorch (backend)
- Pygame (rendering)
- Box2D (physics)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python train.py                # Train (loads previous model and continues)
python demo.py                 # Watch trained agent with boulder and slope
python demo.py untrained       # Watch untrained agent
```

## Training Phases (Transfer Learning)
1. **10M steps** — BipedalWalker-v3 (learn to walk)
2. **20M steps** — SisyphusWalker-v0 with boulder (learn to push)
3. **30M steps** — SisyphusWalker-v0 with boulder + slope (learn to push uphill)

Each phase loads the model from the previous phase and trains further.

## Files
- `config.py` — Settings and hyperparameters
- `train.py` — Training with transfer learning and checkpoints
- `demo.py` — Watch agent visually
- `sisyphus_env.py` — Custom environment with boulder and slope
