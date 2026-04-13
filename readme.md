# Tabula Rasa 🦾
> *"One must imagine Sisyphus happy"* — Albert Camus

A reinforcement learning project where a bipedal agent learns to walk from scratch. Inspired by Camus and the myth of Sisyphus.

**Group 2 — ML2**

## Stack
- Farama Gymnasium — BipedalWalker-v3
- Stable-Baselines3 — PPO
- PyTorch + Matplotlib

## Installation
```bash
pip install -r requirements.txt
```
NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage
```bash
python train.py                                              # Train
python demo.py trained                                       # Watch trained
python demo.py untrained                                     # Watch untrained
python demo.py checkpoint models/checkpoints/<name>          # Watch checkpoint
python demo.py timelapse 10                                  # Timelapse
python plot.py                                               # Learning curve
python visualization.py compare                              # Comparison
```