#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def main():
    setups = [
        "10% (10 ep)",
        "25% (10 ep)",
        "90% (10 ep)",
        "90% (100 ep)",
    ]

    # Dice values from the table
    # Foreground + per-class
    fg_dice =  [0.4289, 0.4323, 0.4337, 0.4901]
    c1_dice =  [0.3317, 0.2302, 0.3460, 0.5872]
    c2_dice =  [0.6481, 0.7038, 0.6481, 0.6456]
    c3_dice =  [0.3071, 0.3628, 0.3071, 0.2376]

    x = np.arange(len(setups))           # positions for groups
    width = 0.18                         # width of each bar

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - 1.5*width, fg_dice, width, label="Foreground Dice")
    ax.bar(x - 0.5*width, c1_dice, width, label="Class 1 Dice")
    ax.bar(x + 0.5*width, c2_dice, width, label="Class 2 Dice")
    ax.bar(x + 1.5*width, c3_dice, width, label="Class 3 Dice")

    ax.set_ylabel("Dice score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(setups, rotation=20)
    ax.set_title("nnU-Net performance across training subsets")
    ax.legend(loc="lower right")

    # Optional: add value labels on top of bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    fig.tight_layout()
    plt.savefig("plt.png")

if __name__ == "__main__":
    main()

