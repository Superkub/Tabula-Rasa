import os
import numpy as np
import matplotlib.pyplot as plt
from config import LOG_PATH

def plot_results():
    f = os.path.join(LOG_PATH, "evaluations.npz")
    if not os.path.exists(f):
        print("No log found — train the model first!")
        return
    data = np.load(f)
    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["results"].mean(axis=1), color="royalblue", linewidth=2)
    plt.title("Tabula Rasa — BipedalWalker Learning Curve", fontsize=14)
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.axhline(y=300, color="green", linestyle="--", label="Solved (300+)")
    plt.axhline(y=0,   color="gray",  linestyle=":",  label="Zero line")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_results()