# Tabula Rasa 🦾
> *"One must imagine Sisyphus happy"* — Albert Camus

A reinforcement learning project where a bipedal agent learns to walk from scratch. Inspired by Camus and the myth of Sisyphus.

**Group 2 — ML2**

## Stack
- Farama Gymnasium — BipedalWalker-v3
- Stable-Baselines3 — PPO
- PyTorch

## Installation
```bash
pip install -r requirements.txt
```
NVIDIA GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage
```bash
python train.py                # Train
python demo.py                 # Watch trained
python demo.py untrained       # Watch untrained
```