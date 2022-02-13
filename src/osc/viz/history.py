"""
Visualization of training history (old).
"""
from collections import defaultdict
from operator import itemgetter

import pandas as pd
from IPython.core.display import Image
from IPython.core.display_functions import display
from matplotlib import pyplot as plt


def viz_history(history):
    duration_sec = max(map(itemgetter("time"), history), default=-1)
    num_steps = max(map(itemgetter("step"), history), default=0)
    print(f"Total training time: {duration_sec/60:.1f} minutes")
    print(f"Average speed: {num_steps/duration_sec:.2f} batches/second")

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    history = history_groupby_name(history)

    for name, ax in zip(["l_global", "l_objects"], axs):
        steps, values = zip(*map(itemgetter("step", "value"), history[f"{name}/train"]))
        values = pd.Series(values).ewm(alpha=0.1).mean().values
        ax.plot(steps, values, label="train")

        steps, values = zip(*map(itemgetter("step", "value"), history[f"{name}/val"]))
        ax.plot(steps, values, label="val")

        ax.set_title(name)
        ax.legend()

    name = "lr"
    ax = axs[-1]
    steps, values = zip(*map(itemgetter("step", "value"), history["lr"]))
    ax.plot(steps, values)
    ax.set_title(name)

    epoch_markers = [h["step"] for h in history["l_global/val"]]
    for ax in axs:
        ax.grid(True, axis="y")
        for e in epoch_markers:
            ax.axvline(e, lw=0.1, color="black")

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig("history.png", dpi=200)
    plt.close(fig)
    display(Image(url="history.png", width=1000))


def history_groupby_name(history):
    history_dict = defaultdict(list)
    for h in history:
        history_dict[h["name"]].append(h)
    return history_dict
